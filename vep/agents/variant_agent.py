import json

from vep.agents.base import accepts
from vep.agents.llm_agent import RECORD_ARG, STOP, TEAM, LLMAgent, send_arg
from vep.validation import check_sequence_pair

# who may send this agent each kind of message
SENDERS = {'task': {'coordinator'}}


class VariantDataAgent(LLMAgent):
    """Checks one assigned variant and passes its approved sequence window on, one record per 'task' message.
    An LLM picks each step: validate_record, get_window and send_message. It may only call validate_record and
    get_window, so it cannot call the model or write reports, and H2 lets its sequences go only to roles that may
    receive them (the model agent). The window stays agent-side: the LLM sees only its length and check result.
    Agent-side checks the LLM cannot skip: no window is sent before validate_record found no errors and the
    window passed its check.
    In:  'task' {'record_id'} from the coordinator.
    Out: 'sequence' {'record_id', 'wt_seq', 'mut_seq'} to the receiver the LLM picks, 'status' to the coordinator:
         done once a window was sent, else failed with the reason codes seen."""
    role = 'variant_data'
    SENDS = 'sequence'
    SYSTEM_PROMPT = TEAM.format(
        role=role, stop=STOP,
        job='the coordinator assigns you one ClinVar variant by its record id. Check the record, fetch its DNA '
            'window and send the checked window to the agent that scores variants. An invalid record must never '
            'be passed on for scoring.')
    TOOLS = {
        'validate_record': ('Check the variant record for errors (alleles, position, DNA window, labels, '
                            'annotations). Returns the list of errors; an empty list means the record is valid.',
                            RECORD_ARG),
        'get_window': ('Fetch the record\'s reference and mutant DNA window and check it. The sequences stay inside '
                       'the tool layer: you get only their length and whether the check passed.', RECORD_ARG),
        'send_message': ('Send the record\'s checked DNA window to another agent. The window goes from the tool '
                         'layer, never through you. Needs a valid record (validate_record with no errors) and its '
                         'window (get_window).', send_arg(role)),
    }

    def __init__(self, agent_id, handle, llm, max_window=8192, max_steps=12):
        super().__init__(agent_id, handle, llm, max_steps)
        self.max_window = max_window  # longest window it will pass on: the approved region size

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        run = self.decide(msg, record_id,
                          f"Message from {msg['from_role']}: task for ClinVar variant record {json.dumps(record_id)}.",
                          valid={}, window={})
        if run['sends']:
            return [*run['sends'], self.status(msg, 'done', record_id=record_id)]
        return [self.status(msg, 'failed', run['codes'] or ['NOT_SENT'], record_id=record_id)]

    def _validate_record(self, rid):
        # re-check at this trust boundary: the record is this agent's responsibility, not the coordinator's
        reply = self._call('validate_record', {'record_id': rid})
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        errors = reply['result']
        self._run['valid'][rid] = not errors
        for e in errors:
            self._code(e['code'])
        return {'ok': True, 'valid': not errors, 'errors': errors}

    def _get_window(self, rid):
        reply = self._call('get_window', {'record_id': rid}, 'sequence')
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        window = reply['result']
        problem = check_sequence_pair(window, self.max_window)
        if problem:  # a malformed or oversized window must not reach the model
            return self._fail(problem)
        self._run['window'][rid] = window
        return {'ok': True, 'window_bp': len(window['wt_seq']), 'check': 'PASSED'}

    def _message(self, rid, to_role):
        if rid not in self._run['valid']:
            return self._fail('NOT_VALIDATED')
        if not self._run['valid'][rid]:
            return self._fail('RECORD_INVALID')
        if rid not in self._run['window']:
            return self._fail('NO_WINDOW')
        window = self._run['window'][rid]
        return self._send(to_role, {'record_id': rid, 'wt_seq': window['wt_seq'], 'mut_seq': window['mut_seq']})
