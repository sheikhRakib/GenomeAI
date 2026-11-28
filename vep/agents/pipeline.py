import json
import os
import time

from tqdm import tqdm

from vep.agents.coordinator import CoordinatorAgent
from vep.agents.base import Runtime
from vep.agents.evidence_agent import EvidenceAgent
from vep.agents.model_agent import ModelAgent
from vep.agents.report_agent import ReportAgent
from vep.agents.single_agent import SingleAgent
from vep.agents.variant_agent import VariantDataAgent
from vep.baseline import score_metrics
from vep.schemas import to_dict
from vep.security import build_security
from vep.tools import ProtectedTools, build_tools
from vep.validation import DNA_RUN

# Agent pipeline (orchestrator side): builds the security stack, the protected tools and the agents,
# sends the dataset to them in bounded tasks, and saves reports, audit log and a run summary.

AGENTS = ((CoordinatorAgent, 'coordinator'), (VariantDataAgent, 'variant_data'), (ModelAgent, 'model'),
          (EvidenceAgent, 'evidence'), (ReportAgent, 'report'))
MESSAGES_PER_RECORD = 11
MODES = ('multi_agent', 'single_agent')


def run_agent_pipeline(df, name, adapter, out_dir, security='mock', batch=100, mode='multi_agent', llm=None):
    """Run every row of df through the agents; return the run summary.
    mode: 'multi_agent' for the five agents, 'single_agent' for the single-agent baseline: one SingleAgent with
    every tool. Every agent is driven by llm (a LangChain chat model with bind_tools, e.g. vep.llm.build_llm).
    out_dir (new) gets reports/<task_id>/<record_id>.json, audit.jsonl, scores.csv, summary.json and
    llm_trace.jsonl: per record and agent, the LLM's tool calls, their results and its replies.
    security: 'permissive' | 'mock' | {'h1'..'h4': ...}, as vep.security.build_security takes it."""
    if mode not in MODES:
        raise ValueError(f'unknown agent mode {mode!r}, expected one of {MODES}')
    if llm is None:
        raise ValueError(f'{mode} mode needs an LLM: every agent decides its tool calls with it')
    os.makedirs(out_dir)  # new folder per run: never overwrites an earlier run
    ids = [int(x) for x in df['ID']] if 'ID' in df else [int(i) for i in df.index]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{name}: record ids are not unique, the agents address records by id')
    records = {rid: row for rid, (_, row) in zip(ids, df.iterrows())}
    sec = build_security(security, audit_path=os.path.join(out_dir, 'audit.jsonl'))
    task_of = {}  # record id -> task id, so build_report gets that subtask's trace

    def context(record_id):
        """Orchestrator-side report fields for one record: its subtask's message trace and H2 decisions."""
        sid = f'{task_of[record_id]}/{record_id}'
        return {'dataset': name,
                'trace': [t for t in runtime.trace if t['subtask_id'] == sid],
                'security_decisions': [to_dict(e.decision) for e in sec.h4.events
                                       if e.request.subtask_id == sid and e.decision is not None
                                       and e.execution_result.get('outcome') != 'pending']}

    tools = ProtectedTools(sec, build_tools(adapter, records, os.path.join(out_dir, 'reports'), context))
    if mode == 'multi_agent':
        agents = [cls(f'{role}-1', tools.for_agent(f'{role}-1', role), llm) for cls, role in AGENTS]
        lead = agents[0]
        lead.max_batch = batch
        # one record costs 11 messages (3 tasks, sequence, score, evidence, result, 4 statuses); 2x headroom still stops loops
        runtime = Runtime(tools, agents, max_messages=MESSAGES_PER_RECORD * 2 * batch + 1)
    else:
        lead = SingleAgent('single-1', tools.for_agent('single-1', 'single'), llm)
        agents = [lead]
        runtime = Runtime(tools, agents, max_messages=2)  # one task message per record; 2x headroom still stops loops

    start = time.perf_counter()
    with tqdm(total=len(ids), desc='Multi-agent' if mode == 'multi_agent' else 'Single agent', unit='variant') as bar:
        for n, first in enumerate(range(0, len(ids), batch)):
            chunk, task_id = ids[first:first + batch], f'task{n:04d}'
            task_of.update(dict.fromkeys(chunk, task_id))
            if mode == 'multi_agent':
                runtime.run('coordinator', 'task', {'record_ids': chunk}, task_id, f'{task_id}/plan')
                bar.update(len(chunk))
                continue
            for rid in chunk:  # no coordinator here: the orchestrator opens one subtask per record, bound to it
                sid = f'{task_id}/{rid}'
                tools.bind_subtask(sid, rid)
                runtime.run('single', 'task', {'record_id': rid}, task_id, sid)
                bar.update(1)
    seconds = time.perf_counter() - start
    if mode == 'multi_agent':
        outcome = {rid: rec for t in lead.tasks for rid, rec in lead.summary(t)['records'].items()}
        for rec in outcome.values():  # the task is over: a record still running never gets its report
            if rec['status'] == 'running':
                rec['status'], rec['reason_codes'] = 'failed', rec['reason_codes'] + ['NO_RESULT']
        task_status = {t: lead.summary(t)['status'] for t in lead.tasks}
    else:  # a record the agent left without an outcome (crash, refused task) failed
        outcome = {rid: lead.records.get(rid, {'status': 'failed', 'reason_codes': ['NO_RESULT']}) for rid in ids}
        task_status = dict.fromkeys(task_of.values(), 'finished')
    _save_traces(out_dir, ids, task_of, agents)
    llm_info = {'llm': lead.llm_name,
                'llm_tool_calls': sum('tool' in s for a in agents for steps in a.traces.values() for s in steps)}
    return _save(df, ids, task_of, name, out_dir, outcome, task_status, lead.anomalies, runtime, sec, adapter,
                 security, mode, seconds, llm_info)


def _save_traces(out_dir, ids, task_of, agents):
    """llm_trace.jsonl: per record and agent, in the order the agents worked on it, what the LLM got, its tool
    calls, the results it read and its replies."""
    with open(os.path.join(out_dir, 'llm_trace.jsonl'), 'x') as f:
        for rid in ids:
            for agent in agents:
                if rid not in agent.traces:
                    continue
                line = json.dumps({'record_id': rid, 'task_id': task_of[rid], 'agent_id': agent.agent_id,
                                   'steps': agent.traces[rid]}, default=str)
                if DNA_RUN.search(line):  # the agents redact already; a sequence must never reach a log
                    raise RuntimeError(f'sequence text in the LLM trace of record {rid}')
                f.write(line + '\n')


def _save(df, ids, task_of, name, out_dir, records, task_status, anomalies, runtime, sec, adapter, security, mode,
          seconds, llm_info):
    """Per-record scores.csv (score read back from each written report) and summary.json.
    records: record id -> {'status', 'reason_codes'}; anomalies: messages the lead agent ignored."""
    status, reasons, paths, scores = [], [], [], []
    for rid in ids:
        rec = records.get(rid, {'status': 'missing', 'reason_codes': [], 'result': None})
        state, codes, path, score = rec['status'], list(rec['reason_codes']), None, None
        if state == 'done':
            # only where the report agent was told to write it, never a path an agent names
            path = os.path.join(out_dir, 'reports', task_of[rid], f'{rid}.json')
            if os.path.isfile(path):
                with open(path) as f:
                    score = json.load(f)['prediction']['score']
            else:
                state, codes, path = 'failed', codes + ['REPORT_MISSING'], None
        status.append(state)
        reasons.append(';'.join(codes))
        paths.append(path)
        scores.append(score)
    cols = [c for c in ('ID', 'CHROM', 'POS', 'REF', 'ALT', 'gene_name', 'labels') if c in df]
    table = df[cols].assign(neg_delta_ll=scores, status=status, reason_codes=reasons, report=paths)
    table.to_csv(os.path.join(out_dir, 'scores.csv'), index=False)

    done = table[table['status'] == 'done']
    outcomes = [e.execution_result.get('outcome') for e in sec.h4.events]
    summary = {'dataset': name, 'mode': mode, 'security': security, 'rows': len(df),
               'done': len(done), 'failed': int((table['status'] == 'failed').sum()),
               'failed_rows': {str(i): r for i, r, s in zip(ids, reasons, status) if s != 'done'},
               'tasks': task_status, 'model': adapter.metadata,
               'evo2_forward_passes': getattr(adapter, 'forward_passes', None),  # run total, counted by the adapter
               'audit_events': len(outcomes), 'denied_or_blocked': sum(o in ('denied', 'blocked') for o in outcomes),
               'instruction_like_alerts': sum('INSTRUCTION_LIKE_CONTENT' in e.execution_result.get('alerts', [])
                                              for e in sec.h4.events),
               'messages': len(runtime.trace), 'undelivered': len(runtime.undelivered),
               'agent_errors': len(runtime.errors),
               'coordinator_anomalies': len(anomalies),  # in single-agent mode, the single agent's
               **llm_info,
               'seconds': round(seconds, 3),
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               **score_metrics(done, done['neg_delta_ll'].astype(float).to_numpy(), name)}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"{name}: {mode} done={summary['done']} failed={summary['failed']} | results saved to {out_dir}")
    return summary
