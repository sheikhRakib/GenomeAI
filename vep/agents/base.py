from collections import deque

from vep.security import RECEIVE_SCOPES

# Message delivered to Agent.receive (built by Runtime, never by the sending agent):
#   {'task_id', 'subtask_id', 'from_id', 'from_role', 'to_role', 'classification', 'body'}
# body has already passed H1 inside the 'transfer' tool; it may be the string '[QUARANTINED]'.


def send(to_role, classification, body, subtask_id=None):
    """What an agent returns from receive(): a request for the Runtime to transfer body to to_role.
    subtask_id opens a new subtask; the Runtime honors it only from the coordinator."""
    out = {'to_role': to_role, 'classification': classification, 'body': body}
    if subtask_id is not None:
        out['subtask_id'] = subtask_id
    return out


class Agent:
    """One identity, one role, one AgentHandle. Subclasses set `role` and implement receive()."""
    role = None

    def __init__(self, agent_id, handle):
        if handle.agent_role != self.role or handle.agent_id != agent_id:
            raise ValueError(f'handle is for {handle.agent_id!r}/{handle.agent_role!r}, '
                             f'not {agent_id!r}/{self.role!r}')
        self.agent_id, self._handle = agent_id, handle

    def call(self, action, arguments, msg, data_classification='none'):
        """Protected tool call inside the subtask the incoming message belongs to."""
        return self._handle.call(action, arguments, msg['task_id'], msg['subtask_id'], data_classification)

    def status(self, msg, state, reason_codes=(), **extra):
        return send('coordinator', 'status', {'agent_id': self.agent_id, 'subtask_id': msg['subtask_id'],
                                              'status': state, 'reason_codes': list(reason_codes), **extra})

    def receive(self, msg):
        """Handle one message; return a list of send(...) requests."""
        raise NotImplementedError


class Runtime:
    """Message bus between agents. Every agent-to-agent message goes through the sender's own
    'transfer' tool call (H2 checks the receiver's data scope, H1 checks the content, H4 audits it),
    so agents never pass objects to each other directly."""
    # shortcut: one agent per role in one process, run agents in separate processes when the interfaces settle

    def __init__(self, tools, agents, max_messages=1000):
        self._tools = tools
        self._agents, self._ids = {}, {}  # role -> agent, role -> agent id fixed at registration
        for a in agents:
            if a.role in self._agents:
                raise ValueError(f'two agents for role {a.role!r}')
            self._agents[a.role], self._ids[a.role] = a, a.agent_id
        self.max_messages = max_messages  # stops agents ping-ponging forever
        self.trace = []  # one entry per message: who sent what to whom, and the security outcome
        self.undelivered = []  # messages to roles with no agent here (e.g. final results to 'user')
        self.errors = []  # agents that crashed or returned malformed requests

    def run(self, to_role, classification, body, task_id, subtask_id):
        """Deliver the user's entry message and every message it causes; return undelivered messages.
        H1 checks the user's message first. When the task's messages are all handled (or the run fails),
        every capability issued in its subtasks is revoked, so none can be replayed later."""
        subtasks = {subtask_id}
        try:
            entry = self._tools.accept_user_message(to_role, body, task_id, subtask_id, classification)
            if not entry['ok']:
                self.errors.append({'task_id': task_id, 'agent_id': 'user', 'reason_codes': entry['reason_codes']})
                return self.undelivered
            queue = deque([{'task_id': task_id, 'subtask_id': subtask_id, 'from_id': 'user', 'from_role': 'user',
                            'to_role': to_role, 'classification': classification, 'body': entry['result']}])
            delivered = 0
            while queue:
                msg = queue.popleft()
                role = msg['to_role']
                agent = self._agents.get(role)
                if agent is None:
                    self.undelivered.append(msg)
                    continue
                delivered += 1
                if delivered > self.max_messages:
                    raise RuntimeError(f'more than {self.max_messages} messages in task {task_id!r}')
                for new in self._deliver(role, agent, msg):
                    subtasks.add(new['subtask_id'])
                    queue.append(new)
        finally:
            for sid in subtasks:
                self._tools.end_subtask(sid)
        return self.undelivered

    def _deliver(self, role, agent, msg):
        """Let one agent handle one message; return the messages it sent that were delivered.
        A crash or malformed request fails only this agent's part of the subtask, never the whole task."""
        agent_id = self._ids[role]
        self._tools.set_active(agent_id, msg['task_id'], msg['subtask_id'])  # its tool calls must be for this message
        try:
            outs = list(agent.receive(msg) or [])  # list(): a generator runs here, while the subtask is active
            bad = [o for o in outs if not (isinstance(o, dict) and isinstance(o.get('to_role'), str)
                                           and isinstance(o.get('classification'), str) and 'body' in o)]
            error = f'{len(bad)} malformed send request(s)' if bad else None
        except Exception as e:
            outs, error = [], f'{type(e).__name__}: {e}'
        finally:
            self._tools.clear_active(agent_id)
        if error:
            self.errors.append({'task_id': msg['task_id'], 'subtask_id': msg['subtask_id'], 'agent_id': agent_id,
                                'error': error})
            if role == 'coordinator':
                return []
            outs = [send('coordinator', 'status', {'agent_id': agent_id, 'subtask_id': msg['subtask_id'],
                                                   'status': 'failed', 'reason_codes': ['AGENT_ERROR']})]
        return [new for out in outs for new in self._transfer(role, msg, out)]

    def _transfer(self, role, msg, out):
        """Run one send() request as the sending agent's own transfer call; [] if it was refused."""
        # sender identity comes from registration, never from the agent's own (changeable) attributes;
        # only the coordinator may open a new subtask, which is then tied to the record it names
        agent_id, sid = self._ids[role], msg['subtask_id']
        if role == 'coordinator' and isinstance(out.get('subtask_id'), str) and out['subtask_id'] != sid:
            sid = out['subtask_id']
            record = out['body'].get('record_id') if isinstance(out['body'], dict) else None
            if not self._tools.bind_subtask(sid, record):
                self.errors.append({'task_id': msg['task_id'], 'subtask_id': sid, 'agent_id': agent_id,
                                    'error': 'subtask already tied to another record'})
                return []
        reply = self._tools.transfer(agent_id, role, out['to_role'], out['body'], msg['task_id'], sid,
                                     out['classification'])
        self.trace.append({'task_id': msg['task_id'], 'subtask_id': sid, 'from_id': agent_id,
                           'from_role': role, 'to_role': out['to_role'],
                           'classification': out['classification'], 'ok': reply['ok'],
                           'reason_codes': reply['reason_codes'], 'request_id': reply['request_id']})
        if not reply['ok']:
            return []
        return [{'task_id': msg['task_id'], 'subtask_id': sid, 'from_id': agent_id,
                 'from_role': role, 'to_role': out['to_role'], 'classification': out['classification'],
                 'body': reply['result']}]


def accepts(role, msg, senders):
    """Reason code if `role` must not act on msg, else None.
    senders: classification -> roles allowed to send it to this agent."""
    if msg['classification'] not in RECEIVE_SCOPES.get(role, ()):
        return 'CLASSIFICATION_NOT_RECEIVABLE'
    if msg['from_role'] not in senders.get(msg['classification'], ()):
        return 'UNEXPECTED_SENDER'
    if not isinstance(msg['body'], dict):  # includes '[QUARANTINED]'
        return 'BAD_MESSAGE'
    return None
