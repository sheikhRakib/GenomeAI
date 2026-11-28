import json
import math

from vep.agents.base import accepts
from vep.agents.llm_agent import RECORD_ARG, STOP, TEAM, LLMAgent, send_arg

PARTS = {'score', 'evidence'}
# who may send this agent each kind of message
SENDERS = {'task': {'coordinator'}, 'score': {'model'}, 'evidence': {'evidence'}}


class ReportAgent(LLMAgent):
    """Joins the parts the coordinator assigned for one subtask (score, evidence) into a report and writes it.
    Each part is checked in code and held; then an LLM picks the next step: build_report, write_report and
    send_message, or wait for the next part. Agent-side checks the LLM cannot skip: no report is built before
    every expected part is held, written before it is built, or announced before it is written.
    It may only call build_report and write_report: it cannot fetch data or rerun the model, the report is
    built server-side from the record, the stored model score and the model's own metadata, evidence must match
    what get_evidence returned for the record, and write_report writes only that built, H1-checked report, as a
    new file inside the approved output dir.
    In:  'task' {'record_id', 'expects'} from the coordinator; 'score' from model; 'evidence' from evidence.
    Out: 'result' {'record_id', 'path'} to the receiver the LLM picks, and 'status' to the coordinator: done
         once the report is written, failed if every part is held and no report was written."""
    role = 'report'
    SENDS = 'result'
    SYSTEM_PROMPT = TEAM.format(
        role=role,
        job='the coordinator assigns you one ClinVar variant, and the parts of its report (its model score and its '
            'evidence) arrive one message at a time. Once every part is held, build the report, write it and send '
            'the result to the agent that tracks the task. While a part is still missing, wait.',
        stop=STOP.replace('DONE, or', 'DONE, WAITING (a part is still missing), or'))
    TOOLS = {
        'build_report': ('Build the result report from the stored model score and the evidence you hold. Needs '
                         'every part of the report.', RECORD_ARG),
        'write_report': ('Write the built report to the approved output folder. Needs build_report first.',
                         RECORD_ARG),
        'send_message': ('Tell another agent that the report is written and where. Needs write_report first.',
                         send_arg(role)),
    }
    # shortcut: a part that arrives before its coordinator task is refused, which the FIFO Runtime rules out; buffer parts if delivery order ever changes

    def __init__(self, agent_id, handle, llm, max_steps=12):
        super().__init__(agent_id, handle, llm, max_steps)
        self._jobs = {}  # (task_id, subtask_id) -> {'record_id', 'expects', 'parts'}

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        key = (msg['task_id'], msg['subtask_id'])
        if msg['classification'] == 'task':
            return self._assign(key, msg, record_id)
        job = self._jobs.get(key)
        if job is None:  # nobody assigned this subtask: refuse unrelated work
            return [self.status(msg, 'failed', ['UNASSIGNED_SUBTASK'], record_id=record_id)]
        kind = msg['classification']
        problem = ('RECORD_MISMATCH' if record_id != job['record_id'] else
                   'UNEXPECTED_PART' if kind not in job['expects'] else
                   'DUPLICATE_PART' if kind in job['parts'] else
                   self._check_part(kind, msg['body']))
        if problem:  # fail closed: this subtask gets no report
            del self._jobs[key]
            return [self.status(msg, 'failed', [problem], record_id=job['record_id'])]
        job['parts'][kind] = msg['body']
        rid = job['record_id']
        run = self.decide(msg, rid,
                          f"Message from {msg['from_role']}: the {kind} part of ClinVar variant record {json.dumps(rid)}. "
                          f"Parts held: {', '.join(sorted(job['parts']))}. "
                          f"A report needs: {', '.join(sorted(job['expects']))}.",
                          job=job, built=False, written=None)
        if run['written']:
            del self._jobs[key]
            return [*run['sends'], self.status(msg, 'done', record_id=rid)]
        if set(job['parts']) == job['expects']:  # no part is left to wait for: fail closed
            del self._jobs[key]
            return [self.status(msg, 'failed', [c for c in run['codes'] if c != 'MISSING_PARTS'] or ['NO_REPORT'],
                                record_id=rid)]
        return []  # the LLM waits for the next part

    def _assign(self, key, msg, record_id):
        expects = msg['body'].get('expects')
        if key in self._jobs:
            return [self.status(msg, 'failed', ['DUPLICATE_TASK'], record_id=record_id)]
        if (isinstance(record_id, bool) or not isinstance(record_id, (int, str)) or not isinstance(expects, list)
                or not expects or not set(expects) <= PARTS):
            return [self.status(msg, 'failed', ['BAD_TASK'], record_id=record_id)]
        self._jobs[key] = {'record_id': record_id, 'expects': set(expects), 'parts': {}}
        return []

    @staticmethod
    def _check_part(kind, body):
        """Reason code if a part is malformed, else None. Values are checked, never changed."""
        if kind == 'score':
            score = body.get('score')
            return None if isinstance(score, float) and math.isfinite(score) else 'BAD_SCORE'
        ok = isinstance(body.get('evidence'), dict) and isinstance(body.get('quarantined'), list)
        return None if ok else 'BAD_EVIDENCE'

    def _build_report(self, rid):
        job = self._run['job']
        if rid != job['record_id']:
            return self._fail('RECORD_MISMATCH')
        missing = sorted(job['expects'] - set(job['parts']))
        if missing:
            # the part comes in a later message, from its own agent: nothing here can fetch it
            return {**self._fail('MISSING_PARTS'), 'missing': missing,
                    'next': 'The missing part arrives in a later message. Reply WAITING now.'}
        evidence = job['parts']['evidence']['evidence'] if 'evidence' in job['parts'] else {}
        # the score and the written report come from the tool side, so this agent cannot change either
        reply = self._call('build_report', {'record_id': rid, 'evidence': evidence}, 'result')
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        self._run['built'] = True
        return {'ok': True, 'built': True, 'score': reply['result']['prediction']['score']}

    def _write_report(self, rid):
        if rid != self._run['job']['record_id']:
            return self._fail('RECORD_MISMATCH')
        if not self._run['built']:
            return self._fail('NOT_BUILT')
        # the orchestrator reads reports only from <task_id>/<record_id>.json, so the agent never picks the path
        reply = self._call('write_report', {'record_id': rid, 'filename': f"{self._run['msg']['task_id']}/{rid}.json"},
                           'result')
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        self._run['written'] = reply['result']
        return {'ok': True, 'written': True}

    def _message(self, rid, to_role):
        if rid != self._run['job']['record_id']:
            return self._fail('RECORD_MISMATCH')
        if not self._run['written']:
            return self._fail('NOT_WRITTEN')
        return self._send(to_role, {'record_id': rid, 'path': self._run['written']})
