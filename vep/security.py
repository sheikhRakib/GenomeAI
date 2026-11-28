import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone

from vep.schemas import Capability, SecurityDecision, hash_arguments, to_dict

# Hook interfaces (student modules implement the same method names):
#   H1 assess(content, source, agent_id) -> {classification, injection_risk, confidence, evidence_span}
#   H2 authorize(request) -> SecurityDecision
#   H3 enforce(request, capability) -> (ok, reason_code)
#   H4 record(event) -> None

POLICY_VERSION = 'mock-policy-0.1'
TTL_SECONDS = 60

POLICY = {
    'coordinator': {'validate_record'},
    'variant_data': {'validate_record', 'get_window'},
    'model': {'score_variant'},
    'evidence': {'get_evidence'},
    'report': {'build_report', 'write_report'},
}
# single-agent baseline (--single-agent): one role holds every tool, to show what splitting the roles adds
POLICY['single'] = set().union(*POLICY.values())

# data classifications each role may receive through 'transfer'
RECEIVE_SCOPES = {
    'coordinator': {'task', 'status', 'result'},  # 'task' is accepted only from the user, see CoordinatorAgent
    'variant_data': {'task'},
    'model': {'sequence'},  # only the window variant_data sends: it takes no task, so none can reach it
    'evidence': {'task'},
    'report': {'task', 'score', 'evidence', 'result'},
    'single': {'task'},  # from the user only, see SingleAgent
}

# keys each message class may carry; any other key could smuggle data past the receiver's class scope
MESSAGE_FIELDS = {
    'task': {'record_id', 'expects'},
    'sequence': {'record_id', 'wt_seq', 'mut_seq'},
    'score': {'record_id', 'score', 'score_name', 'direction', 'uncertainty', 'model'},
    'evidence': {'record_id', 'evidence', 'quarantined'},
    'status': {'agent_id', 'subtask_id', 'status', 'reason_codes', 'record_id'},
    'result': {'record_id', 'path'},
}

INSTRUCTION_LIKE = re.compile(
    r'ignore (all|previous|the) |call (the )?\w+ tool|system prompt|you must|instead,? ', re.I)


def _now():
    return datetime.now(timezone.utc)


def _sign(key, capability):
    return hmac.new(key, capability.signing_payload(), hashlib.sha256).hexdigest()


def issue_capability(request, decision, key):
    """Signed one-request Capability from an allowing decision; constraints come from the decision."""
    if not decision.is_allowed or decision.request_id != request.request_id:
        raise ValueError('decision does not allow this request')
    # no TTL means no lifetime: expires at once (fail closed)
    expires = _now() + timedelta(seconds=decision.ttl_seconds or 0)
    cap = Capability(agent_id=request.agent_id, task_id=request.task_id, subtask_id=request.subtask_id,
                     action=request.requested_action, resource=request.tool_or_resource,
                     constraints={'argument_bounds': dict(decision.argument_bounds),
                                  'data_classification': request.data_classification,
                                  'delegation_depth': decision.delegation_depth,
                                  'granted_scope': list(decision.granted_scope),
                                  'policy_version': decision.policy_version},
                     expires_at=expires.isoformat(),
                     delegation_chain=[])  # no agent-to-agent delegation in the demo yet
    cap.signature = _sign(key, cap)
    return cap


# ---- H1 content trust ----

class PermissiveH1:
    def assess(self, content, source, agent_id):
        return {'classification': 'data', 'injection_risk': 0.0, 'confidence': 1.0, 'evidence_span': None}


class MockH1:
    def assess(self, content, source, agent_id):
        m = INSTRUCTION_LIKE.search(str(content))
        if m:
            return {'classification': 'instruction_like', 'injection_risk': 1.0, 'confidence': 1.0,
                    'evidence_span': m.group(0)}
        # shortcut: fixed regex list misses paraphrased injections, replace with a student H1 classifier
        return {'classification': 'data', 'injection_risk': 0.0, 'confidence': 0.6, 'evidence_span': None}


# ---- H2 authorization ----

class PermissiveH2:
    def authorize(self, request):
        return SecurityDecision(request_id=request.request_id, decision='allow',
                                granted_scope=[request.tool_or_resource], ttl_seconds=TTL_SECONDS,
                                reason_codes=['PERMISSIVE_ALLOW'], method_version='permissive-0.1',
                                policy_version='permissive')


class MockH2:
    def authorize(self, request):
        role, action = request.agent_role, request.requested_action
        if role not in POLICY:
            reasons = ['UNKNOWN_ROLE']
        elif action == 'transfer':
            to_role = request.arguments.get('to_role')
            if to_role not in RECEIVE_SCOPES:
                reasons = ['UNKNOWN_RECEIVER']
            elif request.data_classification not in RECEIVE_SCOPES[to_role]:
                reasons = ['CLASSIFICATION_NOT_RECEIVABLE']
            else:
                reasons = []
        elif action not in POLICY[role]:
            reasons = ['ACTION_NOT_PERMITTED']
        else:
            reasons = []
        common = {'request_id': request.request_id, 'method_version': 'mock-0.1',
                  'policy_version': POLICY_VERSION}
        if reasons:
            return SecurityDecision(decision='deny', reason_codes=reasons, risk_score=1.0, **common)
        # bounds = exact requested arguments, so any later change is caught by H3
        return SecurityDecision(decision='allow_with_constraints', granted_scope=[request.tool_or_resource],
                                argument_bounds=dict(request.arguments), ttl_seconds=TTL_SECONDS,
                                delegation_depth=0, reason_codes=['POLICY_ALLOW'], **common)


# ---- H3 execution enforcement ----

class PermissiveH3:
    """Checks signature, expiry, revocation only: same pipeline as mock, no scope or bounds."""

    def __init__(self, key, revoked):
        self._key, self._revoked = key, revoked

    def enforce(self, request, capability):
        if capability is None:
            return False, 'CAP_MISSING'
        if not isinstance(capability, Capability) or not isinstance(capability.signature, str):
            return False, 'CAP_INVALID'
        if not hmac.compare_digest(capability.signature, _sign(self._key, capability)):
            return False, 'CAP_BAD_SIGNATURE'
        try:
            expired = datetime.fromisoformat(capability.expires_at) <= _now()
        except (TypeError, ValueError):
            return False, 'CAP_INVALID'
        if expired:
            return False, 'CAP_EXPIRED'
        if capability.capability_id in self._revoked:
            return False, 'CAP_REVOKED'
        return True, 'CAP_OK'


class MockH3(PermissiveH3):
    def __init__(self, key, revoked):
        super().__init__(key, revoked)
        self._used = set()

    def enforce(self, request, capability):
        ok, code = super().enforce(request, capability)
        if not ok:
            return ok, code
        if capability.capability_id in self._used:
            return False, 'CAP_REPLAYED'
        for name, want, got in (('AGENT', capability.agent_id, request.agent_id),
                                ('TASK', capability.task_id, request.task_id),
                                ('SUBTASK', capability.subtask_id, request.subtask_id),
                                ('ACTION', capability.action, request.requested_action),
                                ('RESOURCE', capability.resource, request.tool_or_resource)):
            if want != got:
                return False, f'CAP_{name}_MISMATCH'
        c = capability.constraints
        if c.get('data_classification') != request.data_classification:
            return False, 'CAP_SCOPE_MISMATCH'
        if request.arguments and hash_arguments(request.arguments) != request.argument_hash:
            return False, 'ARG_HASH_MISMATCH'
        bounds = c.get('argument_bounds', {})
        if set(request.arguments) - set(bounds):
            return False, 'ARG_UNBOUNDED'
        # canonical-JSON hash, not ==: catches 1 / 1.0 / True swaps
        if hash_arguments(bounds) != hash_arguments(request.arguments):
            return False, 'ARG_OUT_OF_BOUNDS'
        if len(capability.delegation_chain) > c.get('delegation_depth', 0):
            return False, 'DELEGATION_TOO_DEEP'
        self._used.add(capability.capability_id)  # one-time use: blocks replay
        return True, 'CAP_OK'


# ---- H4 audit ----

class AuditLog:
    def __init__(self, path=None):
        self.events, self.path = [], path

    def record(self, event):
        self.events.append(event)
        if self.path:
            with open(self.path, 'a') as f:
                f.write(json.dumps(to_dict(event), default=str) + '\n')


MODES = {'permissive': (PermissiveH1, PermissiveH2, PermissiveH3, AuditLog),
         'mock': (MockH1, MockH2, MockH3, AuditLog)}
HOOKS = ('h1', 'h2', 'h3', 'h4')
METHODS = {'h1': 'assess', 'h2': 'authorize', 'h3': 'enforce', 'h4': 'record'}


class SecurityStack:
    """h1..h4 hooks plus the capability key; agents never get this object."""

    def __init__(self):
        self._key = secrets.token_bytes(32)  # per process, never leaves the security layer
        self._revoked = set()

    def issue_capability(self, request, decision):
        return issue_capability(request, decision, self._key)

    def revoke(self, capability_id):
        self._revoked.add(capability_id)


def build_security(config, audit_path=None):
    """config: 'permissive' | 'mock' for all hooks, or {'h1'..'h4': mode name or hook object}.
    Unknown names, unknown keys, or a missing hook raise ValueError: never falls back to permissive."""
    if isinstance(config, str):
        config = dict.fromkeys(HOOKS, config)
    if not isinstance(config, dict) or set(config) != set(HOOKS):
        raise ValueError(f'security config needs exactly the keys {HOOKS}, got {config!r}')
    stack = SecurityStack()
    for i, hook in enumerate(HOOKS):
        spec = config[hook]
        if isinstance(spec, str):
            if spec not in MODES:
                raise ValueError(f'unknown security module {spec!r} for {hook}')
            cls = MODES[spec][i]
            obj = cls(stack._key, stack._revoked) if hook == 'h3' else cls(audit_path) if hook == 'h4' else cls()
        else:
            obj = spec
            if not callable(getattr(obj, METHODS[hook], None)):
                raise ValueError(f'{hook} object has no {METHODS[hook]}()')
            # a custom H3 that verifies our signatures gets the key through bind()
            if hook == 'h3' and callable(getattr(obj, 'bind', None)):
                obj.bind(stack._key, stack._revoked)
        setattr(stack, hook, obj)
    return stack
