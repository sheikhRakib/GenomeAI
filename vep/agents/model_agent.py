import math

from vep.agents.base import Agent, accepts, send
from vep.validation import check_sequence_pair

# who may send this agent each kind of message
SENDERS = {'sequence': {'variant_data'}}


class ModelAgent(Agent):
    """Scores one variant per 'sequence' message with the frozen model behind the score_variant tool.
    In:  {'record_id', 'wt_seq', 'mut_seq'} from the variant/data agent.
    Out: 'score' message to the report agent, 'status' message to the coordinator."""
    role = 'model'

    def __init__(self, agent_id, handle, max_window=8192):
        super().__init__(agent_id, handle)
        self.max_window = max_window  # longest window it will score: cannot expand the approved region

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS) or check_sequence_pair(msg['body'], self.max_window)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        body = msg['body']
        # the tool scores only this record's approved window, so a substituted pair is refused there too
        reply = self.call('score_variant', {'record_id': record_id, 'wt_seq': body['wt_seq'],
                                            'mut_seq': body['mut_seq']}, msg, 'sequence')
        if not reply['ok']:
            return [self.status(msg, 'failed', reply['reason_codes'], record_id=record_id)]
        out = reply['result']
        score = out.get('score') if isinstance(out, dict) else None
        if not isinstance(score, float) or not math.isfinite(score) or not isinstance(out.get('model'), dict):
            # malformed model output must not reach the report
            return [self.status(msg, 'failed', ['BAD_MODEL_OUTPUT'], record_id=record_id)]
        result = {'record_id': record_id, 'score': score, 'score_name': 'neg_delta_ll',
                  'direction': 'higher = less likely under the model = more disruptive',
                  # shortcut: zero-shot score is not a probability, add calibration (e.g. percentile vs benign set) when needed
                  'uncertainty': None, 'model': out['model']}
        return [send('report', 'score', result), self.status(msg, 'done', record_id=record_id)]
