import json
import math

from vep.agents.base import accepts
from vep.agents.llm_agent import RECORD_ARG, STOP, TEAM, LLMAgent, send_arg
from vep.validation import check_sequence_pair

# who may send this agent each kind of message
SENDERS = {'sequence': {'variant_data'}}


class ModelAgent(LLMAgent):
    """Scores one variant per 'sequence' message with the frozen model behind the score_variant tool.
    An LLM picks each step: score_variant and send_message. The window from the message stays agent-side: the LLM
    sees only its record id and length, and score_variant scores only that window.
    Agent-side checks the LLM cannot skip: the message's window is checked before the LLM sees the message, no
    other record is scored, and only a finite model score is sent on.
    In:  'sequence' {'record_id', 'wt_seq', 'mut_seq'} from the variant/data agent.
    Out: 'score' message to the receiver the LLM picks, 'status' message to the coordinator."""
    role = 'model'
    SENDS = 'score'
    SYSTEM_PROMPT = TEAM.format(
        role=role, stop=STOP,
        job='you get the checked DNA window of one ClinVar variant. Score it and send the score to the agent '
            'that writes the report.')
    TOOLS = {
        'score_variant': ('Score the variant with the Evo2 genomic foundation model on the DNA window you received: '
                          'neg_delta_ll = log-likelihood of the reference window minus that of the mutant window; '
                          'higher means more disruptive. It is not a probability.', RECORD_ARG),
        'send_message': ('Send the record\'s model score to another agent. Needs score_variant first.',
                         send_arg(role)),
    }

    def __init__(self, agent_id, handle, llm, max_window=8192, max_steps=12):
        super().__init__(agent_id, handle, llm, max_steps)
        self.max_window = max_window  # longest window it will score: cannot expand the approved region

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS) or check_sequence_pair(msg['body'], self.max_window)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        body = msg['body']
        run = self.decide(msg, record_id,
                          f"Message from {msg['from_role']}: the checked DNA window of ClinVar variant record "
                          f"{json.dumps(record_id)} ({len(body['wt_seq'])} bp; the sequences stay in the tool layer).",
                          window={record_id: {'wt_seq': body['wt_seq'], 'mut_seq': body['mut_seq']}}, score={})
        if run['sends']:
            return [*run['sends'], self.status(msg, 'done', record_id=record_id)]
        return [self.status(msg, 'failed', run['codes'] or ['NOT_SENT'], record_id=record_id)]

    def _score_variant(self, rid):
        if rid not in self._run['window']:  # only the window this message carried
            return self._fail('NO_WINDOW')
        # the tool scores only this record's approved window, so a substituted pair is refused there too
        reply = self._call('score_variant', {'record_id': rid, **self._run['window'][rid]}, 'sequence')
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        out = reply['result']
        score = out.get('score') if isinstance(out, dict) else None
        if not isinstance(score, float) or not math.isfinite(score) or not isinstance(out.get('model'), dict):
            return self._fail('BAD_MODEL_OUTPUT')  # malformed model output must not reach the report
        self._run['score'][rid] = {
            'record_id': rid, 'score': score, 'score_name': 'neg_delta_ll',
            'direction': 'higher = less likely under the model = more disruptive',
            # shortcut: zero-shot score is not a probability, add calibration (e.g. percentile vs benign set) when needed
            'uncertainty': None, 'model': out['model']}
        return {'ok': True, 'score': score, 'score_name': 'neg_delta_ll', 'direction': 'higher = more disruptive',
                'model': out['model'].get('model')}

    def _message(self, rid, to_role):
        if rid not in self._run['score']:
            return self._fail('NO_SCORE')
        return self._send(to_role, self._run['score'][rid])
