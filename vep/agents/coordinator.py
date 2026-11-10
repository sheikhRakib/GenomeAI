import copy
import re

from vep.agents.base import Agent, accepts, send

REASON_CODE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')  # status may carry codes only, never data

# who may send this agent each kind of message
SENDERS = {'task': {'user'},
           'status': {'variant_data', 'model', 'evidence', 'report'},
           'result': {'report'}}
# the task graph for one variant: who gets a 'task' message, and what it carries besides the record id
DISPATCH = (('variant_data', {}),                              # window -> model agent -> score -> report
            ('evidence', {}),                                  # evidence -> report
            ('report', {'expects': ['score', 'evidence']}))    # joins both, then writes the report
FINISHER = 'report'  # a subtask is done when this role reports done


class CoordinatorAgent(Agent):
    """Validates the user task, opens one subtask per valid record, dispatches it, and tracks status.
    It may only call validate_record and receive task/status/result, so it never sees sequences,
    scores or evidence.
    In:  'task' {'record_ids': [...]} from the user; 'status' from the other agents; 'result' from report.
    Out: 'task' messages to variant_data, evidence and report, one subtask per record."""
    role = 'coordinator'

    def __init__(self, agent_id, handle, max_batch=100):
        super().__init__(agent_id, handle)
        self.max_batch = max_batch  # bounds how many records (and so model calls) one task may ask for
        self.tasks = {}  # task_id -> {'status', 'reason_codes', 'records': {record_id: {...}}, 'subtasks': {sid: record_id}}
        self.anomalies = []  # messages ignored, with the reason

    def receive(self, msg):
        problem = accepts(self.role, msg, SENDERS)
        if problem:
            return self._ignore(msg, problem)
        if msg['classification'] == 'task':
            return self._plan(msg)
        return self._track(msg)

    def summary(self, task_id):
        """Copy of a task's state: status 'rejected' | 'running' | 'finished', and per-record results."""
        task = self.tasks.get(task_id)
        if task is None:
            return None
        states = [r['status'] for r in task['records'].values()]
        status = task['status'] if task['status'] == 'rejected' else 'running' if 'running' in states else 'finished'
        return {'task_id': task_id, 'status': status, 'reason_codes': list(task['reason_codes']),
                'records': copy.deepcopy(task['records'])}

    def _plan(self, msg):
        task_id = msg['task_id']
        if task_id in self.tasks:
            return self._ignore(msg, 'TASK_EXISTS')
        task = self.tasks[task_id] = {'status': 'accepted', 'reason_codes': [], 'records': {}, 'subtasks': {}}
        ids = msg['body'].get('record_ids')
        problem = self._check_task(ids)
        if problem:
            task['status'], task['reason_codes'] = 'rejected', [problem]
            return []
        out = []
        for rid in ids:
            reply = self.call('validate_record', {'record_id': rid}, msg)
            codes = (reply['reason_codes'] if not reply['ok']
                     else [e['code'] for e in reply['result']] if reply['result'] else [])
            if codes:  # invalid or unknown record: never dispatched
                task['records'][rid] = {'subtask_id': None, 'status': 'failed', 'reason_codes': codes,
                                        'agents': {}, 'result': None}
                continue
            sid = f'{task_id}/{rid}'
            task['subtasks'][sid] = rid
            task['records'][rid] = {'subtask_id': sid, 'status': 'running', 'reason_codes': [],
                                    'agents': {}, 'result': None}
            out += [send(role, 'task', {'record_id': rid, **extra}, subtask_id=sid) for role, extra in DISPATCH]
        return out

    def _check_task(self, ids):
        """Reason code if the requested record list is not a small, unique list of ids, else None."""
        if not isinstance(ids, list):
            return 'BAD_TASK'
        if not ids:
            return 'EMPTY_TASK'
        if len(ids) > self.max_batch:
            return 'BATCH_TOO_LARGE'
        if any(isinstance(i, bool) or not isinstance(i, (int, str)) for i in ids):
            return 'BAD_RECORD_ID'
        if len(set(ids)) != len(ids):
            return 'DUPLICATE_RECORDS'
        return None

    def _track(self, msg):
        task = self.tasks.get(msg['task_id'])
        rid = task['subtasks'].get(msg['subtask_id']) if task else None
        if rid is None:
            return self._ignore(msg, 'UNKNOWN_SUBTASK')
        rec, body = task['records'][rid], msg['body']
        if msg['classification'] == 'result':
            rec['result'] = True  # only that a report was written: the orchestrator finds it by record, not by path
            return []
        state, codes = body.get('status'), body.get('reason_codes')
        if state not in ('done', 'failed'):
            return self._ignore(msg, 'BAD_STATUS')
        rec['agents'][msg['from_id']] = state  # sender identity from the runtime, not from the body
        if state == 'failed':
            rec['status'] = 'failed'
            codes = codes if isinstance(codes, list) else []
            clean = [c for c in codes if isinstance(c, str) and REASON_CODE.match(c)]
            if len(clean) != len(codes):  # anything else could smuggle data into the run summary
                clean.append('BAD_REASON_CODE')
            rec['reason_codes'] += [f"{msg['from_role']}:{c}" for c in clean]
        elif msg['from_role'] == FINISHER and rec['status'] == 'running':
            rec['status'] = 'done'  # a failure reported earlier is never overwritten
        return []

    def _ignore(self, msg, reason):
        self.anomalies.append({'task_id': msg['task_id'], 'subtask_id': msg['subtask_id'], 'from_id': msg['from_id'],
                               'from_role': msg['from_role'], 'classification': msg['classification'],
                               'reason': reason})
        return []
