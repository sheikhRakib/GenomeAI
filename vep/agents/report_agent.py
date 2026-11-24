import math

from vep.agents.base import Agent, accepts, send

PARTS = {'score', 'evidence'}
# who may send this agent each kind of message
SENDERS = {'task': {'coordinator'}, 'score': {'model'}, 'evidence': {'evidence'}}


class ReportAgent(Agent):
    """Joins the parts the coordinator assigned for one subtask (score, evidence) into a report and writes it.
    It may only call build_report and write_report: it cannot fetch data or rerun the model, the report is
    built server-side from the record, the stored model score and the model's own metadata, evidence must match
    what get_evidence returned for the record, and write_report writes only that built, H1-checked report, as a
    new file inside the approved output dir.
    In:  'task' {'record_id', 'expects'} from the coordinator; 'score' from model; 'evidence' from evidence.
    Out: 'result' {'record_id', 'path'} and 'status' to the coordinator."""
    role = 'report'
    # shortcut: a part that arrives before its coordinator task is refused, which the FIFO Runtime rules out; buffer parts if delivery order ever changes

    def __init__(self, agent_id, handle):
        super().__init__(agent_id, handle)
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
        if set(job['parts']) != job['expects']:
            return []
        del self._jobs[key]
        return self._write(msg, job)

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

    def _write(self, msg, job):
        rid = job['record_id']
        evidence = job['parts']['evidence']['evidence'] if 'evidence' in job['parts'] else {}
        # the score and the written report come from the tool side, so this agent cannot change either
        reply = self.call('build_report', {'record_id': rid, 'evidence': evidence}, msg, 'result')
        if not reply['ok']:
            return [self.status(msg, 'failed', reply['reason_codes'], record_id=rid)]
        reply = self.call('write_report', {'record_id': rid, 'filename': f"{msg['task_id']}/{rid}.json"},
                          msg, 'result')
        if not reply['ok']:
            return [self.status(msg, 'failed', reply['reason_codes'], record_id=rid)]
        return [send('coordinator', 'result', {'record_id': rid, 'path': reply['result']}),
                self.status(msg, 'done', record_id=rid)]
