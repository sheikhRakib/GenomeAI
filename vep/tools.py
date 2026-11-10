import copy
import dataclasses
import hashlib
import json
import math
import os
import time

from vep.evidence import get_evidence
from vep.report import build_report
from vep.validation import check_sequence_pair, validate_record
from vep.schemas import ActionRequest, AuditEvent, SecurityDecision, to_dict

QUARANTINED = '[QUARANTINED]'
# evidence fields copied from the dataset beyond vep.evidence.UNTRUSTED_FIELDS
EXTRA_UNTRUSTED = ['gene', 'clinvar_id']
# actions whose arguments carry message, sequence or report content: the audit keeps only their hash
HASH_ONLY_AUDIT = {'transfer', 'user_message', 'score_variant', 'build_report', 'write_report'}


def build_tools(adapter, records, out_dir='reports', context=None, max_window=8192):
    """Raw tool functions by action name. records: dict record id -> record (dict or pandas row).
    context: optional orchestrator callable, record id -> {'dataset', 'trace', 'security_decisions'} for
    build_report; agents never supply these, so they cannot forge them.
    Scores and built reports stay on the tool side: agents name a record, never supply the score or report.
    Give this only to ProtectedTools, never to agents."""
    scores, built = {}, {}  # record id -> model score / built report

    def get_window(record_id):
        # shortcut: windows come from the dataset's precomputed columns, replace with genome FASTA lookup in normalize/get_window
        r = records[record_id]
        return {'wt_seq': str(r['wt_seq']), 'mut_seq': str(r['mut_seq'])}

    def score_variant(record_id, wt_seq, mut_seq):
        problem = check_sequence_pair({'wt_seq': wt_seq, 'mut_seq': mut_seq}, max_window)
        if problem:
            raise ValueError(problem)
        if {'wt_seq': wt_seq, 'mut_seq': mut_seq} != get_window(record_id):
            raise ValueError('sequences are not the approved window for this record')
        # float(): model may return numpy/torch scalars, which are not JSON for hashing
        score = float(adapter.score_pair(wt_seq, mut_seq))
        if not math.isfinite(score):
            raise ValueError(f'model returned a non-finite score {score}')
        scores[record_id] = score
        # metadata comes from the adapter, not the agent, so score provenance cannot be forged
        return {'score': score, 'model': dict(adapter.metadata)}

    def evidence(record_id):
        ev = get_evidence(records[record_id])
        ev['untrusted_fields'] = ev['untrusted_fields'] + EXTRA_UNTRUSTED
        return ev

    def report(record_id, evidence):
        if record_id not in scores:
            raise ValueError('no model score for this record')
        # trace and security_decisions are not caller arguments: agents could forge them
        built[record_id] = build_report(records[record_id], scores[record_id], adapter.metadata, evidence,
                                        **(context(record_id) if context else {}))
        return built[record_id]

    def write_report(record_id, filename):
        root = os.path.realpath(out_dir)
        path = os.path.realpath(os.path.join(root, filename))
        if os.path.commonpath([root, path]) != root:
            raise ValueError(f'path {filename!r} is outside the approved output dir')
        if record_id not in built:
            raise ValueError('no built report for this record')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'x') as f:  # 'x': never overwrite an existing file (audit log, other reports)
            json.dump(built[record_id], f, indent=2)
        return path
    write_report.out_dir = out_dir  # ProtectedTools checks the audit log is not under it

    return {
        'validate_record': lambda record_id: validate_record(records[record_id]),
        'get_window': get_window,
        'score_variant': score_variant,
        'get_evidence': evidence,
        'build_report': report,
        'write_report': write_report,
        # shortcut: returns the message to the sender's orchestrator for delivery, add a mailbox when agents run apart
        'transfer': lambda to_role, message: message,
    }


def _hash(obj):
    """sha256 of canonical JSON (same as schemas.hash_arguments for JSON values), repr if not JSON."""
    try:
        s = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)
    except (TypeError, ValueError):
        s = repr(obj)
    return hashlib.sha256(s.encode()).hexdigest()


def _plain(obj):
    """Plain JSON copy: subclasses with tricked repr, hash or equality become plain values, and the caller
    keeps no reference it could change after the checks. Raises on anything that is not JSON data."""
    return json.loads(json.dumps(obj))


def _text_paths(obj, path=()):
    """Key paths of every text value (a string, or a list of strings) inside obj."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _text_paths(v, path + (k,))]
    if isinstance(obj, list) and not all(isinstance(v, str) for v in obj):
        return [p for i, v in enumerate(obj) for p in _text_paths(v, path + (i,))]
    return [path] if isinstance(obj, (str, list)) else []


def _untrusted_paths(action, result):
    """Key paths in a tool output holding outside text (dataset annotations, other agents, the user)."""
    if action in ('transfer', 'user_message'):
        return [()]
    if action == 'validate_record':  # error messages quote record values
        return [(i, 'message') for i in range(len(result))]
    if action == 'get_evidence':
        return [(f,) for f in result['untrusted_fields']]
    if action == 'build_report':  # evidence came back through an agent: check every text in it again
        return [('evidence',) + p for p in _text_paths(result['evidence'])]
    return []


def _get_path(obj, path):
    for k in path:
        obj = obj[k]
    return obj


class AgentHandle:
    """What an agent holds: identity is fixed at creation, the agent only picks action and arguments."""

    def __init__(self, tools, agent_id, agent_role):
        self._tools, self.agent_id, self.agent_role = tools, agent_id, agent_role

    def call(self, action, arguments, task_id, subtask_id, data_classification='none'):
        # shortcut: transfer data_classification is caller-declared, derive it from the producing tool's output in the agent layer
        return self._tools._call(self.agent_id, self.agent_role, action, arguments, task_id, subtask_id,
                                 data_classification)


class ProtectedTools:
    """Orchestrator-side: hands out AgentHandles; every call goes H2 -> capability -> H3 -> audit -> tool -> H1 -> H4.
    Before H2 the tool layer also checks that the call belongs to the subtask the agent is handling now and to
    that subtask's record, and that messages between agents go only through the Runtime."""
    # shortcut: in-process isolation only (Python cannot stop a caller reaching private attributes), move agents to separate processes when interfaces settle

    def __init__(self, security, registry):
        out_dir = getattr(registry.get('write_report'), 'out_dir', None)
        audit = getattr(security.h4, 'path', None)
        if out_dir and audit:
            root, audit = os.path.realpath(out_dir), os.path.realpath(audit)
            if os.path.commonpath([root, audit]) == root:
                raise ValueError('audit log must not be inside the report output dir')
        self._security = security
        self._registry = dict(registry)
        self._roles = {}  # agent_id -> role, set once
        self._issued = {}  # subtask_id -> capability ids, revoked at end_subtask
        self._bound = {}  # subtask_id -> the one record its calls may touch, set by the Runtime
        self._active = {}  # agent_id -> (task_id, subtask_id) of the message it is handling, set by the Runtime

    def for_agent(self, agent_id, agent_role):
        if self._roles.setdefault(agent_id, agent_role) != agent_role:
            raise ValueError(f'agent {agent_id!r} already registered as {self._roles[agent_id]!r}')
        return AgentHandle(self, agent_id, agent_role)

    def decisions_for(self, task_id):
        """Copies of the H2 decisions audited for a task, for the orchestrator to put in reports."""
        return [to_dict(e.decision) for e in getattr(self._security.h4, 'events', [])
                if e.request.task_id == task_id and e.decision is not None
                and e.execution_result.get('outcome') != 'pending']

    # ---- Runtime side ----

    def bind_subtask(self, subtask_id, record_id):
        """Tie a subtask to one record; a later different record for the same subtask is refused."""
        return self._bound.setdefault(subtask_id, record_id) == record_id

    def set_active(self, agent_id, task_id, subtask_id):
        self._active[agent_id] = (task_id, subtask_id)

    def clear_active(self, agent_id):
        self._active.pop(agent_id, None)

    def end_subtask(self, subtask_id):
        self._bound.pop(subtask_id, None)
        for capability_id in self._issued.pop(subtask_id, []):
            self._security.revoke(capability_id)

    def transfer(self, agent_id, agent_role, to_role, message, task_id, subtask_id, data_classification):
        """An agent's message, sent by the Runtime under the sender's identity (agents cannot call this)."""
        return self._call(agent_id, agent_role, 'transfer', {'to_role': to_role, 'message': message},
                          task_id, subtask_id, data_classification, internal=True)

    def accept_user_message(self, to_role, message, task_id, subtask_id, data_classification):
        """The user's entry message: H1 checks it before any agent acts on it, H4 audits it."""
        start = time.perf_counter()
        try:
            arguments = _plain({'to_role': to_role, 'message': message})
            request = ActionRequest(task_id=task_id, subtask_id=subtask_id, agent_id='user', agent_role='user',
                                    requested_action='user_message', tool_or_resource=f'agent:{to_role}',
                                    arguments=arguments, data_classification=data_classification,
                                    source_agent='user')
        except Exception as e:
            return self._bad_request('user', 'user', 'user_message', message, task_id, subtask_id,
                                     data_classification, start, e)
        try:
            result, trust, quarantined = self._assess(request, arguments['message'])
        except Exception as e:
            return self._finish(request, None, start, 'failed', ['H1_FAILED'], errors=[repr(e)])
        return self._finish(request, None, start, 'executed', ['USER_MESSAGE'], None, result, trust, quarantined)

    # ---- every call ----

    def _call(self, agent_id, agent_role, action, arguments, task_id, subtask_id, data_classification,
              internal=False):
        start = time.perf_counter()
        try:
            if not isinstance(arguments, dict):
                raise TypeError(f'arguments must be a dict, got {type(arguments).__name__}')
            arguments = _plain(arguments)  # checks, audit and tool all see these exact plain values
            resource = f'agent:{arguments["to_role"]}' if action == 'transfer' else action
            request = ActionRequest(task_id=task_id, subtask_id=subtask_id, agent_id=agent_id,
                                    agent_role=agent_role, requested_action=action, tool_or_resource=resource,
                                    arguments=arguments, data_classification=data_classification,
                                    source_agent=agent_id)
        except Exception as e:
            return self._bad_request(agent_id, agent_role, action, arguments, task_id, subtask_id,
                                     data_classification, start, e)
        problem = self._scope(request, internal)
        if problem:
            return self._finish(request, None, start, 'denied', [problem])
        try:
            decision = self._security.h2.authorize(request)
            if not isinstance(decision, SecurityDecision):
                raise TypeError(f'H2 returned {type(decision).__name__}, not SecurityDecision')
        except Exception as e:
            return self._finish(request, None, start, 'denied', ['H2_FAILED'], errors=[repr(e)])
        if not decision.is_allowed:
            return self._finish(request, decision, start, 'denied', decision.reason_codes or ['DENIED'])
        try:
            capability = self._security.issue_capability(request, decision)
        except Exception as e:
            return self._finish(request, decision, start, 'denied', ['ISSUE_FAILED'], errors=[repr(e)])
        self._issued.setdefault(subtask_id, []).append(capability.capability_id)
        return self._execute(request, decision, capability, start)

    def _scope(self, request, internal):
        """Reason code if the call is outside what this agent is handling now, else None."""
        if request.requested_action == 'transfer' and not internal:
            return 'TRANSFER_VIA_RUNTIME_ONLY'  # a direct transfer would be logged but never delivered
        active = self._active.get(request.agent_id)
        if active is not None and active != (request.task_id, request.subtask_id):
            return 'NOT_CURRENT_SUBTASK'
        bound = self._bound.get(request.subtask_id)
        if bound is not None and 'record_id' in request.arguments and request.arguments['record_id'] != bound:
            return 'RECORD_OUT_OF_SCOPE'
        return None

    def _bad_request(self, agent_id, agent_role, action, arguments, task_id, subtask_id, data_classification,
                     start, error):
        # hash-only request so the bad call is still audited
        request = ActionRequest(task_id=str(task_id), subtask_id=str(subtask_id), agent_id=agent_id,
                                agent_role=agent_role, requested_action=str(action), tool_or_resource=str(action),
                                argument_hash=_hash(arguments), data_classification=str(data_classification),
                                source_agent=agent_id)
        return self._finish(request, None, start, 'denied', ['BAD_REQUEST'], errors=[repr(error)])

    def _enforce_with(self, request, capability, decision):
        """Run a request with a given capability (H3 onward); tests use it for tampered or stale capabilities."""
        start = time.perf_counter()
        if not isinstance(decision, SecurityDecision) or not decision.is_allowed:
            return self._finish(request, None, start, 'denied', ['NO_DECISION'], capability)
        return self._execute(request, decision, capability, start)

    def _execute(self, request, decision, capability, start):
        try:
            ok, code = self._security.h3.enforce(request, capability)
        except Exception as e:
            return self._finish(request, decision, start, 'blocked', ['H3_FAILED'], capability, errors=[repr(e)])
        if ok is not True:
            return self._finish(request, decision, start, 'blocked', [code], capability)
        fn = self._registry.get(request.requested_action)
        if fn is None:
            return self._finish(request, decision, start, 'failed', ['UNKNOWN_TOOL'], capability)
        codes = list(decision.reason_codes) + [code]
        # durable intent before side effects: no audit, no execution
        if not self._record(request, decision, start, 'pending', codes, capability):
            return self._reply(request, False, None, codes + ['AUDIT_FAILED'], None)
        try:
            result = fn(**request.arguments)
        except Exception as e:
            return self._finish(request, decision, start, 'failed', ['TOOL_FAILED'], capability,
                                errors=[f'{type(e).__name__}: {e}'])
        try:
            result, trust, quarantined = self._assess(request, result)
        except Exception as e:
            # untrusted text nobody could assess: withhold the result
            return self._finish(request, decision, start, 'failed', ['H1_FAILED'], capability, errors=[repr(e)])
        return self._finish(request, decision, start, 'executed', codes, capability, result, trust, quarantined)

    def _assess(self, request, result):
        """H1 on every untrusted field; fields not classed 'data' are replaced in a copy of the result."""
        action = request.requested_action
        source = {'transfer': f'agent:{request.source_agent}', 'user_message': 'user'}.get(action, f'tool:{action}')
        trust, quarantined, out = {}, {}, None
        for path in _untrusted_paths(action, result):
            value = _get_path(result, path)
            if value is None:
                continue
            text = ' | '.join(map(str, value)) if isinstance(value, list) else str(value)
            a = self._security.h1.assess(text, source, request.agent_id)
            if not isinstance(a, dict) or 'classification' not in a:
                raise TypeError(f'H1 returned {a!r}')
            name = '.'.join(map(str, path)) or 'message'
            trust[name] = a
            if a['classification'] != 'data':
                quarantined[name] = _hash(value)  # original kept only as hash in the audit
                if not path:
                    out = QUARANTINED
                    continue
                out = copy.deepcopy(result) if out is None else out
                _get_path(out, path[:-1])[path[-1]] = QUARANTINED
        return (result if out is None else out), (trust or None), quarantined

    def _record(self, request, decision, start, outcome, reason_codes, capability=None,
                result_hash=None, trust=None, quarantined=None, errors=()):
        """Write one H4 event; False if it could not be recorded."""
        try:
            if request.requested_action in HASH_ONLY_AUDIT:
                if request.arguments:
                    request = dataclasses.replace(request, arguments={})  # argument_hash stays
                if decision is not None and decision.argument_bounds:
                    # H2 bounds copy the arguments; their hash stays in enforced_constraints
                    decision = dataclasses.replace(decision, argument_bounds={})
            constraints = getattr(capability, 'constraints', None)
            constraints = dict(constraints) if isinstance(constraints, dict) else {}
            bounds = constraints.pop('argument_bounds', None)
            alerts = [] if outcome in ('executed', 'pending') else list(reason_codes)
            if quarantined:
                alerts.append('INSTRUCTION_LIKE_CONTENT')
            summary = {'outcome': outcome, 'reason_codes': list(reason_codes),
                       'capability_id': getattr(capability, 'capability_id', None),
                       'enforced_constraints': {**constraints,
                                                'argument_bounds_hash': None if bounds is None else _hash(bounds)},
                       'result_hash': result_hash, 'content_trust': trust,
                       'quarantined': quarantined or {}, 'alerts': alerts}
            event = AuditEvent(request=request, decision=decision, execution_result=summary,
                               errors=list(errors), latency_ms=(time.perf_counter() - start) * 1000)
            self._security.h4.record(event)
            return True
        except Exception:
            return False

    def _finish(self, request, decision, start, outcome, reason_codes, capability=None,
                result=None, trust=None, quarantined=None, errors=()):
        ok = outcome == 'executed'
        rh = _hash(result) if ok else None
        if not self._record(request, decision, start, outcome, reason_codes, capability, rh, trust,
                            quarantined, errors):
            # unaudited action must not look successful
            ok, reason_codes = False, list(reason_codes) + ['AUDIT_FAILED']
        return self._reply(request, ok, result, reason_codes, trust)

    def _reply(self, request, ok, result, reason_codes, trust):
        return {'ok': ok, 'result': result if ok else None, 'reason_codes': list(reason_codes),
                'content_trust': trust, 'request_id': request.request_id}
