from vep.agents.base import Agent, accepts, send
from vep.validation import check_sequence_pair

# who may send this agent each kind of message
SENDERS = {'task': {'coordinator'}}


class VariantDataAgent(Agent):
    """Checks one assigned variant and fetches its approved sequence window, one record per 'task' message.
    It may only call validate_record and get_window, so it cannot call the model or write reports,
    and H2 lets its sequences go only to roles that may receive them (the model agent).
    In:  'task' {'record_id'} from the coordinator.
    Out: 'sequence' {'record_id', 'wt_seq', 'mut_seq'} to the model agent, 'status' to the coordinator."""
    role = 'variant_data'

    def __init__(self, agent_id, handle, max_window=8192):
        super().__init__(agent_id, handle)
        self.max_window = max_window  # longest window it will pass on: the approved region size

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        # re-check at this trust boundary: the record is this agent's responsibility, not the coordinator's
        reply = self.call('validate_record', {'record_id': record_id}, msg)
        if not reply['ok']:
            return [self.status(msg, 'failed', reply['reason_codes'], record_id=record_id)]
        if reply['result']:
            return [self.status(msg, 'failed', [e['code'] for e in reply['result']], record_id=record_id)]
        reply = self.call('get_window', {'record_id': record_id}, msg, 'sequence')
        if not reply['ok']:
            return [self.status(msg, 'failed', reply['reason_codes'], record_id=record_id)]
        problem = check_sequence_pair(reply['result'], self.max_window)
        if problem:  # a malformed or oversized window must not reach the model
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        window = reply['result']
        return [send('model', 'sequence', {'record_id': record_id, 'wt_seq': window['wt_seq'],
                                           'mut_seq': window['mut_seq']}),
                self.status(msg, 'done', record_id=record_id)]
