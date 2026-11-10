import json
import os

from tqdm import tqdm

from vep.agents.coordinator import CoordinatorAgent
from vep.agents.base import Runtime
from vep.agents.evidence_agent import EvidenceAgent
from vep.agents.model_agent import ModelAgent
from vep.agents.report_agent import ReportAgent
from vep.agents.variant_agent import VariantDataAgent
from vep.baseline import score_metrics
from vep.schemas import to_dict
from vep.security import build_security
from vep.tools import ProtectedTools, build_tools

# Agent pipeline (orchestrator side): builds the security stack, the protected tools and the five agents,
# sends the dataset to the coordinator in bounded tasks, and saves reports, audit log and a run summary.

AGENTS = ((CoordinatorAgent, 'coordinator'), (VariantDataAgent, 'variant_data'), (ModelAgent, 'model'),
          (EvidenceAgent, 'evidence'), (ReportAgent, 'report'))
MESSAGES_PER_RECORD = 11


def run_agent_pipeline(df, name, adapter, out_dir, security='mock', batch=100):
    """Run every row of df through the five agents; return the run summary.
    out_dir (new) gets reports/<task_id>/<record_id>.json, audit.jsonl, scores.csv and summary.json.
    security: 'permissive' | 'mock' | {'h1'..'h4': ...}, as vep.security.build_security takes it."""
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
    agents = [cls(f'{role}-1', tools.for_agent(f'{role}-1', role)) for cls, role in AGENTS]
    coordinator = agents[0]
    coordinator.max_batch = batch
    # one record costs 11 messages (3 tasks, sequence, score, evidence, result, 4 statuses); 2x headroom still stops loops
    runtime = Runtime(tools, agents, max_messages=MESSAGES_PER_RECORD * 2 * batch + 1)

    with tqdm(total=len(ids), desc='Agents', unit='variant') as bar:
        for n, start in enumerate(range(0, len(ids), batch)):
            chunk, task_id = ids[start:start + batch], f'task{n:04d}'
            task_of.update(dict.fromkeys(chunk, task_id))
            runtime.run('coordinator', 'task', {'record_ids': chunk}, task_id, f'{task_id}/plan')
            bar.update(len(chunk))
    return _save(df, ids, task_of, name, out_dir, coordinator, runtime, sec, adapter, security)


def _save(df, ids, task_of, name, out_dir, coordinator, runtime, sec, adapter, security):
    """Per-record scores.csv (score read back from each written report) and summary.json."""
    records = {rid: rec for t in coordinator.tasks for rid, rec in coordinator.summary(t)['records'].items()}
    task_status = {t: coordinator.summary(t)['status'] for t in coordinator.tasks}
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
    summary = {'dataset': name, 'mode': 'agents', 'security': security, 'rows': len(df),
               'done': len(done), 'failed': int((table['status'] == 'failed').sum()),
               'failed_rows': {str(i): r for i, r, s in zip(ids, reasons, status) if s != 'done'},
               'tasks': task_status, 'model': adapter.metadata,
               'audit_events': len(outcomes), 'denied_or_blocked': sum(o in ('denied', 'blocked') for o in outcomes),
               'instruction_like_alerts': sum('INSTRUCTION_LIKE_CONTENT' in e.execution_result.get('alerts', [])
                                              for e in sec.h4.events),
               'messages': len(runtime.trace), 'undelivered': len(runtime.undelivered),
               'agent_errors': len(runtime.errors),
               'coordinator_anomalies': len(coordinator.anomalies),
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               **score_metrics(done, done['neg_delta_ll'].astype(float).to_numpy(), name)}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"{name}: agents done={summary['done']} failed={summary['failed']} | results saved to {out_dir}")
    return summary
