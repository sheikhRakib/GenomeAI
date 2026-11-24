import math

from vep.agents.base import Agent, accepts
from vep.validation import check_sequence_pair

# who may send this agent each kind of message
SENDERS = {'task': {'user'}}


class SingleAgent(Agent):
    """Single-agent baseline: one role runs every step for one record per 'task' message and holds every tool.
    Tools, hooks and agent-side checks are the five-agent pipeline's, so comparing the two isolates what
    splitting the work into least-privilege roles adds. It sends no messages: there is no other agent.
    In:  'task' {'record_id'} from the user (the orchestrator), in a subtask the orchestrator bound to that record.
    Out: nothing; each record's outcome goes to self.records."""
    role = 'single'

    def __init__(self, agent_id, handle, max_window=8192):
        super().__init__(agent_id, handle)
        self.max_window = max_window  # longest window it will score: the approved region size
        self.records = {}  # record_id -> {'status': 'done' | 'failed', 'reason_codes': [...]}
        self.anomalies = []  # messages ignored, with the reason

    def receive(self, msg):
        rid = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = (accepts(self.role, msg, SENDERS)
                   or ('BAD_TASK' if isinstance(rid, bool) or not isinstance(rid, (int, str)) else None)
                   or ('DUPLICATE_TASK' if rid in self.records else None))  # a first outcome is never overwritten
        if problem:
            self.anomalies.append({'task_id': msg['task_id'], 'subtask_id': msg['subtask_id'],
                                   'from_role': msg['from_role'], 'reason': problem})
            return []
        codes = self._process(msg, rid)
        self.records[rid] = {'status': 'failed' if codes else 'done', 'reason_codes': codes}
        return []

    def _process(self, msg, rid):
        """Every step for one record; [] once its report is written, else the reason codes."""
        reply = self.call('validate_record', {'record_id': rid}, msg)
        if not reply['ok']:
            return reply['reason_codes']
        if reply['result']:  # invalid record: never reaches the model
            return [e['code'] for e in reply['result']]
        reply = self.call('get_window', {'record_id': rid}, msg, 'sequence')
        if not reply['ok']:
            return reply['reason_codes']
        window = reply['result']
        problem = check_sequence_pair(window, self.max_window)
        if problem:  # a malformed or oversized window must not reach the model
            return [problem]
        reply = self.call('score_variant', {'record_id': rid, 'wt_seq': window['wt_seq'],
                                            'mut_seq': window['mut_seq']}, msg, 'sequence')
        if not reply['ok']:
            return reply['reason_codes']
        out = reply['result']
        score = out.get('score') if isinstance(out, dict) else None
        if not isinstance(score, float) or not math.isfinite(score) or not isinstance(out.get('model'), dict):
            return ['BAD_MODEL_OUTPUT']
        reply = self.call('get_evidence', {'record_id': rid}, msg, 'evidence')
        if not reply['ok']:
            return reply['reason_codes']
        evidence = reply['result']
        if not isinstance(evidence, dict) or not evidence.get('source'):
            return ['BAD_EVIDENCE']
        # the score and the written report come from the tool side, as in the five-agent pipeline
        reply = self.call('build_report', {'record_id': rid, 'evidence': evidence}, msg, 'result')
        if not reply['ok']:
            return reply['reason_codes']
        reply = self.call('write_report', {'record_id': rid, 'filename': f"{msg['task_id']}/{rid}.json"},
                          msg, 'result')
        return [] if reply['ok'] else reply['reason_codes']
