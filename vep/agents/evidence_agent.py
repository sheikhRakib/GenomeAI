from vep.agents.base import Agent, accepts, send

# who may send this agent each kind of message
SENDERS = {'task': {'coordinator'}}


class EvidenceAgent(Agent):
    """Fetches public annotation evidence for one assigned variant per 'task' message and passes it on as data.
    It may only call get_evidence: no side-effecting tools. Annotation text is never read as instructions;
    fields H1 flagged arrive as '[QUARANTINED]' and are listed so the report can say so.
    In:  'task' {'record_id'} from the coordinator.
    Out: 'evidence' {'record_id', 'evidence', 'quarantined'} to the report agent, 'status' to the coordinator."""
    role = 'evidence'

    def receive(self, msg):
        record_id = msg['body'].get('record_id') if isinstance(msg['body'], dict) else None
        problem = accepts(self.role, msg, SENDERS)
        if problem:
            return [self.status(msg, 'failed', [problem], record_id=record_id)]
        reply = self.call('get_evidence', {'record_id': record_id}, msg, 'evidence')
        if not reply['ok']:
            return [self.status(msg, 'failed', reply['reason_codes'], record_id=record_id)]
        evidence = reply['result']
        if not isinstance(evidence, dict) or not evidence.get('source'):
            return [self.status(msg, 'failed', ['BAD_EVIDENCE'], record_id=record_id)]
        quarantined = sorted(k for k, a in (reply['content_trust'] or {}).items() if a['classification'] != 'data')
        return [send('report', 'evidence', {'record_id': record_id, 'evidence': evidence, 'quarantined': quarantined}),
                self.status(msg, 'done', ['EVIDENCE_QUARANTINED'] if quarantined else [], record_id=record_id)]
