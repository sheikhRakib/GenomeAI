import copy
import json
import re

from vep.agents.base import accepts
from vep.agents.llm_agent import RECORD_ARG, STOP, TEAM, LLMAgent, send_arg

REASON_CODE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')  # status may carry codes only, never data

# who may send this agent each kind of message
SENDERS = {'task': {'user'},
           'status': {'variant_data', 'model', 'evidence', 'report'},
           'result': {'report'}}
REPORT_PARTS = ['score', 'evidence']  # what a report is made of: a 'task' to report says it waits for both
FINISHER = 'report'  # a subtask is done when this role reports done


class CoordinatorAgent(LLMAgent):
    """Checks the user task, then for each record an LLM decides: it calls validate_record and picks which agents
    get a 'task' message for it (send_message), or stops. Each record it assigns gets its own subtask. Status and
    result messages are only recorded, in code: tracking makes no decision.
    It may only call validate_record and receive task/status/result, so it never sees sequences, scores or
    evidence. Agent-side checks the LLM cannot skip: the task's size and ids are checked before the LLM sees it,
    the LLM works on one record at a time and only on it, and no record is assigned before validate_record found
    no errors.
    In:  'task' {'record_ids': [...]} from the user; 'status' from the other agents; 'result' from report.
    Out: 'task' messages to the receivers the LLM picks, one subtask per record."""
    role = 'coordinator'
    SENDS = 'task'
    SYSTEM_PROMPT = TEAM.format(
        role=role, stop=STOP,
        job='the user sends a task of ClinVar variant records, and you get one record of it at a time. Validate '
            'the record, then assign a valid record to every agent that must work on it, so that it ends with a '
            'written report: its score, its evidence and the report that joins them. Never assign an invalid record.')
    TOOLS = {
        'validate_record': ('Check the variant record for errors (alleles, position, DNA window, labels, '
                            'annotations). Returns the list of errors; an empty list means the record is valid.',
                            RECORD_ARG),
        'send_message': ('Assign the record to another agent: it gets a task message for this record. Needs a '
                         'valid record (validate_record with no errors).', send_arg(role)),
    }

    def __init__(self, agent_id, handle, llm, max_batch=100, max_steps=12):
        super().__init__(agent_id, handle, llm, max_steps)
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
        for rid in ids:  # one LLM run per record, so a large task never fills its context
            run = self.decide(msg, rid, f"Message from {msg['from_role']}: task {json.dumps(task_id)} of "
                                        f"{len(ids)} records; plan ClinVar variant record {json.dumps(rid)}.",
                              valid={})
            if not run['sends']:  # invalid, unknown or not assigned: never dispatched
                task['records'][rid] = {'subtask_id': None, 'status': 'failed',
                                        'reason_codes': run['codes'] or ['NOT_ASSIGNED'], 'agents': {}, 'result': None}
                continue
            sid = f'{task_id}/{rid}'
            task['subtasks'][sid] = rid
            task['records'][rid] = {'subtask_id': sid, 'status': 'running', 'reason_codes': [],
                                    'agents': {}, 'result': None}
            out += run['sends']
        return out

    def _validate_record(self, rid):
        if rid != self._run['rid']:  # the plan subtask is not bound to a record, so this agent keeps the scope
            return self._fail('RECORD_OUT_OF_SCOPE')
        reply = self._call('validate_record', {'record_id': rid})
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        errors = reply['result']
        self._run['valid'][rid] = not errors
        for e in errors:
            self._code(e['code'])
        return {'ok': True, 'valid': not errors, 'errors': errors}

    def _message(self, rid, to_role):
        if rid != self._run['rid']:
            return self._fail('RECORD_OUT_OF_SCOPE')
        if rid not in self._run['valid']:
            return self._fail('NOT_VALIDATED')
        if not self._run['valid'][rid]:
            return self._fail('RECORD_INVALID')
        body = {'record_id': rid, **({'expects': list(REPORT_PARTS)} if to_role == 'report' else {})}
        return self._send(to_role, body, subtask_id=f"{self._run['msg']['task_id']}/{rid}")

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
