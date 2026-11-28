import json

from vep.agents.base import accepts
from vep.agents.llm_agent import RECORD_ARG, STOP, TEAM, LLMAgent, send_arg

# who may send this agent each kind of message
SENDERS = {'task': {'coordinator'}}


class EvidenceAgent(LLMAgent):
    """Fetches public annotation evidence for one assigned variant per 'task' message and passes it on as data.
    An LLM picks each step: get_evidence and send_message. It may only call get_evidence: no side-effecting tools.
    Annotation text is never read as instructions; fields H1 flagged arrive as '[QUARANTINED]' and are listed so
    the report can say so. Agent-side checks the LLM cannot skip: only evidence get_evidence returned, with a
    source, is sent on.
    In:  'task' {'record_id'} from the coordinator.
    Out: 'evidence' {'record_id', 'evidence', 'quarantined'} to the receiver the LLM picks, 'status' to the
         coordinator."""
    role = 'evidence'
    SENDS = 'evidence'
    SYSTEM_PROMPT = TEAM.format(
        role=role, stop=STOP,
        job='the coordinator assigns you one ClinVar variant by its record id. Fetch its annotation evidence and '
            'send it to the agent that writes the report.')
    TOOLS = {
        'get_evidence': ('Retrieve public ClinVar annotation evidence for the record: clinical significance, '
                         'condition, HGVS name, allele frequency. Fields H1 flagged as instruction-like read '
                         '[QUARANTINED].', RECORD_ARG),
        'send_message': ('Send the record\'s evidence, as get_evidence returned it, to another agent. Needs '
                         'get_evidence first.', send_arg(role)),
    }

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        run = self.decide(msg, record_id,
                          f"Message from {msg['from_role']}: task for ClinVar variant record {json.dumps(record_id)}.",
                          evidence={})
        if not run['sends']:
            return [self.status(msg, 'failed', run['codes'] or ['NOT_SENT'], record_id=record_id)]
        quarantined = any(s['body']['quarantined'] for s in run['sends'])
        return [*run['sends'],
                self.status(msg, 'done', ['EVIDENCE_QUARANTINED'] if quarantined else [], record_id=record_id)]

    def _get_evidence(self, rid):
        reply = self._call('get_evidence', {'record_id': rid}, 'evidence')
        if not reply['ok']:
            return self._fail(*reply['reason_codes'])
        evidence = reply['result']
        if not isinstance(evidence, dict) or not evidence.get('source'):
            return self._fail('BAD_EVIDENCE')
        quarantined = sorted(k for k, a in (reply['content_trust'] or {}).items() if a['classification'] != 'data')
        self._run['evidence'][rid] = {'record_id': rid, 'evidence': evidence, 'quarantined': quarantined}
        return {'ok': True, 'evidence': evidence}

    def _message(self, rid, to_role):
        if rid not in self._run['evidence']:
            return self._fail('NO_EVIDENCE')
        return self._send(to_role, self._run['evidence'][rid])
