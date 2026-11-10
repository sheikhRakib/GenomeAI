import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Common interface objects every security module exchanges (spec section 4).
# Bump SCHEMA_VERSION on any field change: from_dict rejects other versions so modules fail closed.
SCHEMA_VERSION = '0.1'

DECISIONS = {'allow', 'deny', 'allow_with_constraints'}


def _id():
    return uuid.uuid4().hex


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def hash_arguments(arguments):
    """sha256 hex of the arguments as canonical JSON (key order does not matter)."""
    return hashlib.sha256(_canonical(arguments).encode()).hexdigest()


@dataclass
class ActionRequest:
    task_id: str
    subtask_id: str
    agent_id: str
    agent_role: str
    requested_action: str
    tool_or_resource: str
    arguments: dict = field(default_factory=dict)
    argument_hash: str = ''
    data_classification: str = ''
    source_agent: str = ''
    provenance: list = field(default_factory=list)
    risk_context: dict = field(default_factory=dict)
    request_id: str = field(default_factory=_id)
    timestamp: str = field(default_factory=_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if not self.argument_hash:
            self.argument_hash = hash_arguments(self.arguments)
        # hash-only requests (empty arguments) are allowed; otherwise hash must match the arguments
        elif self.arguments and self.argument_hash != hash_arguments(self.arguments):
            raise ValueError('argument_hash does not match arguments')


@dataclass
class SecurityDecision:
    request_id: str
    decision: str
    granted_scope: list = field(default_factory=list)
    argument_bounds: dict = field(default_factory=dict)
    ttl_seconds: int | None = None
    delegation_depth: int = 0
    required_approval: str = ''
    reason_codes: list = field(default_factory=list)
    risk_score: float = 0.0
    method_version: str = ''
    policy_version: str = ''
    decision_id: str = field(default_factory=_id)
    timestamp: str = field(default_factory=_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.decision not in DECISIONS:
            raise ValueError(f'unknown decision {self.decision!r}, expected one of {sorted(DECISIONS)}')

    @property
    def is_allowed(self):
        # an allow that still waits on approval must not proceed
        return self.decision in ('allow', 'allow_with_constraints') and not self.required_approval


@dataclass
class Capability:
    # data only: signing and verification belong to the security module that issues it
    agent_id: str
    task_id: str
    subtask_id: str
    action: str
    resource: str
    constraints: dict = field(default_factory=dict)
    expires_at: str = ''
    delegation_chain: list = field(default_factory=list)
    capability_id: str = field(default_factory=_id)
    signature: str = ''
    schema_version: str = SCHEMA_VERSION

    def signing_payload(self):
        """Canonical JSON bytes of every field except signature, for HMAC or similar."""
        d = dataclasses.asdict(self)
        del d['signature']
        return _canonical(d).encode()


@dataclass
class AuditEvent:
    request: ActionRequest
    decision: SecurityDecision | None = None
    execution_result: Any = None
    changed_arguments: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    latency_ms: float | None = None
    model_version: str = ''
    tool_version: str = ''
    prior_event_ids: list = field(default_factory=list)
    event_id: str = field(default_factory=_id)
    timestamp: str = field(default_factory=_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        # from_dict passes nested objects as dicts: rebuild them so they are schema-checked too
        if isinstance(self.request, dict):
            self.request = from_dict(ActionRequest, self.request)
        if isinstance(self.decision, dict):
            self.decision = from_dict(SecurityDecision, self.decision)


def to_dict(obj):
    return dataclasses.asdict(obj)


def from_dict(cls, d):
    """Build cls from a dict; raise ValueError on unknown keys or a different schema_version."""
    unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f'{cls.__name__}: unknown keys {sorted(unknown)}')
    if d.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'{cls.__name__}: schema_version {d.get("schema_version")!r} != {SCHEMA_VERSION!r}')
    return cls(**d)
