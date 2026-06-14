"""Anggira RL — Reinforcement Learning from Scratch

Phase 4: Reinforcement Learning
- GridWorld MDP (4×4 grid, terminal state)
- Dynamic Programming (Policy Evaluation, Policy Iteration, Value Iteration)
- Q-Learning and SARSA (Tabular TD Control with ε-greedy)
- DQN (Deep Q-Network with Experience Replay and Target Network)
- PPO (Proximal Policy Optimization with Clipped Surrogate & GAE)

All pure Python — no numpy. DQN uses a hand-made neural net (list-of-lists).
"""

import math
import random
from collections import deque


# ═══════════════════════════════════════════════════════════════
#  ACTIVATIONS & HELPERS
# ═══════════════════════════════════════════════════════════════

def _relu(x):
    return max(0.0, x)


def _relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def _softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _argmax(lst):
    """Return index of first maximum."""
    best_val = lst[0]
    best_idx = 0
    for i, v in enumerate(lst):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def _text_plot(values, width=30):
    """Compact text-based line plot (every N-th point)."""
    if not values:
        return "(empty)"
    if len(values) <= width:
        sampled = values
    else:
        step = max(1, len(values) // width)
        sampled = [values[min(i * step, len(values) - 1)] for i in range(width)]
    mn, mx = min(sampled), max(sampled)
    if mx - mn < 1e-12:
        return "─" * len(sampled)
    out = []
    for v in sampled:
        p = int((v - mn) / (mx - mn) * 5)  # 0-5 range for compact chars
        out.append(["▁", "▂", "▃", "▄", "▅", "█"][p])
    return "".join(out)


# ═══════════════════════════════════════════════════════════════
#  GRIDWORLD MDP — 4×4 grid
# ═══════════════════════════════════════════════════════════════

ACTIONS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class GridWorld:
    """4×4 gridworld.

    States 0-15, state 15 (bottom-right) is terminal.
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    Default reward: -1 per step, 0 at terminal.
    """

    N_STATES = 16
    N_ACTIONS = 4
    TERMINAL = 15
    SIZE = 4

    def __init__(self, reward_default=-1.0, reward_terminal=0.0):
        self.reward_default = reward_default
        self.reward_terminal = reward_terminal
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, state, action):
        """Return (next_state, reward, done)."""
        if state == self.TERMINAL:
            return self.TERMINAL, self.reward_terminal, True
        row, col = divmod(state, self.SIZE)
        if action == 0:      # UP
            row -= 1
        elif action == 1:    # DOWN
            row += 1
        elif action == 2:    # LEFT
            col -= 1
        elif action == 3:    # RIGHT
            col += 1
        row = max(0, min(self.SIZE - 1, row))
        col = max(0, min(self.SIZE - 1, col))
        ns = row * self.SIZE + col
        if ns == self.TERMINAL:
            return ns, self.reward_terminal, True
        return ns, self.reward_default, False

    def is_terminal(self, state):
        return state == self.TERMINAL

    def render_policy(self, policy):
        """Render policy as arrows on grid."""
        grid = []
        for r in range(self.SIZE):
            row_chars = []
            for c in range(self.SIZE):
                s = r * self.SIZE + c
                if s == self.TERMINAL:
                    row_chars.append("G")
                else:
                    row_chars.append(ACTIONS.get(policy[s], "?"))
            grid.append("  ".join(row_chars))
        return "\n".join(grid)

    def render_values(self, values, fmt="{:.2f}"):
        """Render state values on grid."""
        lines = []
        for r in range(self.SIZE):
            row_vals = []
            for c in range(self.SIZE):
                s = r * self.SIZE + c
                if s == self.TERMINAL:
                    row_vals.append(" TERM  ")
                else:
                    row_vals.append(fmt.format(values[s]).rjust(6))
            lines.append(" ".join(row_vals))
        return "\n".join(lines)

    def all_states(self):
        """Return non-terminal states."""
        return [s for s in range(self.N_STATES) if s != self.TERMINAL]


# ═══════════════════════════════════════════════════════════════
#  DYNAMIC PROGRAMMING
# ═══════════════════════════════════════════════════════════════

def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    """Iterative policy evaluation. Returns state-value array length N_STATES."""
    V = [0.0] * env.N_STATES
    while True:
        delta = 0.0
        for s in env.all_states():
            v_old = V[s]
            a = policy[s]
            ns, r, _ = env.step(s, a)
            V[s] = r + gamma * V[ns]
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def policy_iteration(env, gamma=0.9):
    """Policy iteration → optimal policy + optimal values."""
    policy = [0] * env.N_STATES  # initial arbitrary policy (all UP)
    while True:
        V = policy_evaluation(env, policy, gamma)
        stable = True
        for s in env.all_states():
            old_a = policy[s]
            best_a, best_v = 0, -float("inf")
            for a in range(env.N_ACTIONS):
                ns, r, _ = env.step(s, a)
                v = r + gamma * V[ns]
                if v > best_v:
                    best_v = v
                    best_a = a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        if stable:
            break
    return policy, V


def value_iteration(env, gamma=0.9, theta=1e-6):
    """Value iteration → optimal values + optimal policy."""
    V = [0.0] * env.N_STATES
    while True:
        delta = 0.0
        for s in env.all_states():
            v_old = V[s]
            best = -float("inf")
            for a in range(env.N_ACTIONS):
                ns, r, _ = env.step(s, a)
                best = max(best, r + gamma * V[ns])
            V[s] = best
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    # Extract greedy policy
    policy = [0] * env.N_STATES
    for s in env.all_states():
        best_a, best_v = 0, -float("inf")
        for a in range(env.N_ACTIONS):
            ns, r, _ = env.step(s, a)
            v = r + gamma * V[ns]
            if v > best_v:
                best_v = v
                best_a = a
        policy[s] = best_a
    return policy, V


# ═══════════════════════════════════════════════════════════════
#  Q-LEARNING  (Tabular, Off-Policy TD Control)
# ═══════════════════════════════════════════════════════════════

class QLearning:
    """Q-Learning with ε-greedy exploration."""

    def __init__(self, n_states=16, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = [[0.0] * n_actions for _ in range(n_states)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return _argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        best_next = max(self.Q[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def greedy_policy(self):
        return [_argmax(self.Q[s]) for s in range(len(self.Q))]


# ═══════════════════════════════════════════════════════════════
#  SARSA  (Tabular, On-Policy TD Control)
# ═══════════════════════════════════════════════════════════════

class SARSA:
    """SARSA with ε-greedy exploration."""

    def __init__(self, n_states=16, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = [[0.0] * n_actions for _ in range(n_states)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return _argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, next_action, done):
        q_next = self.Q[next_state][next_action] if not done else 0.0
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def greedy_policy(self):
        return [_argmax(self.Q[s]) for s in range(len(self.Q))]


# ═══════════════════════════════════════════════════════════════
#  MLP — Minimal Neural Network (list-of-lists matrices)
#  Used by both DQN and PPO
# ═══════════════════════════════════════════════════════════════

class MLP:
    """Multi-layer perceptron built with list-of-lists.

    layer_sizes: [input_dim, hidden1, ..., output_dim]
    Hidden activations: ReLU. Output: linear.
    Supports forward, backward (MSE grad), and custom grad.
    """

    def __init__(self, layer_sizes, lr=0.001):
        self.num_layers = len(layer_sizes) - 1
        self.lr = lr
        self.weights = []   # list of matrices [out_dim × in_dim]
        self.biases = []    # list of bias vectors
        for i in range(self.num_layers):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            scale = math.sqrt(2.0 / n_in)  # He init
            w = [[random.gauss(0, scale) for _ in range(n_in)] for _ in range(n_out)]
            b = [0.0] * n_out
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass. x is a list of floats. Returns output list."""
        self._inputs = [x]
        self._zs = []
        self._activations = [x]   # activations before each layer (input is first)
        for li in range(self.num_layers):
            w = self.weights[li]
            b = self.biases[li]
            inp = self._activations[-1]
            z = [sum(w[j][k] * inp[k] for k in range(len(inp))) + b[j]
                 for j in range(len(w))]
            self._zs.append(z)
            # ReLU for hidden layers, linear for last
            if li < self.num_layers - 1:
                out = [_relu(v) for v in z]
            else:
                out = z  # linear output
            self._activations.append(out)
        return self._activations[-1]

    def backward_mse(self, target, predicted):
        """Backward pass for MSE loss: L = mean((pred - target)²).

        Computes gradients and updates weights via SGD.
        """
        # dL/d(output)
        d_out = [2.0 * (p - t) / len(predicted) for p, t in zip(predicted, target)]
        self._backward_from_grad(d_out)

    def backward_custom(self, grad_output):
        """Backward pass with custom output gradient (for PPO actor)."""
        self._backward_from_grad(grad_output)

    def _backward_from_grad(self, grad_output):
        """Generic backward from gradient w.r.t. output. Updates weights."""
        grad = grad_output[:]
        for li in range(self.num_layers - 1, -1, -1):
            w = self.weights[li]
            inp = self._activations[li]
            z = self._zs[li]

            # Gradient through activation
            if li < self.num_layers - 1:
                dz = [grad[j] * _relu_derivative(z[j]) for j in range(len(grad))]
            else:
                dz = grad  # linear output — identity

            # Weight & bias gradients
            # dw[j][k] = dz[j] * inp[k]
            for j in range(len(w)):
                self.biases[li][j] -= self.lr * dz[j]
                for k in range(len(w[j])):
                    w[j][k] -= self.lr * dz[j] * inp[k]

            # Backprop gradient to previous layer
            if li > 0:
                new_grad = [0.0] * len(inp)
                for k in range(len(inp)):
                    for j in range(len(w)):
                        new_grad[k] += dz[j] * w[j][k]
                grad = new_grad

    def copy_from(self, other):
        """Copy weights and biases from another MLP of the same architecture."""
        for li in range(self.num_layers):
            for j in range(len(self.weights[li])):
                self.biases[li][j] = other.biases[li][j]
                for k in range(len(self.weights[li][j])):
                    self.weights[li][j][k] = other.weights[li][j][k]

    def zero_grad(self):
        pass  # Gradients are computed and applied inline


# ═══════════════════════════════════════════════════════════════
#  REPLAY BUFFER  (for DQN)
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════
#  DQN — Deep Q-Network
# ═══════════════════════════════════════════════════════════════

class DQN:
    """Deep Q-Network with experience replay and target network.

    Uses a hand-made MLP with list-of-lists (no numpy).
    """

    def __init__(self, n_states=16, n_actions=4, hidden_dim=16,
                 lr=0.01, gamma=0.9, epsilon=0.1, epsilon_min=0.01,
                 epsilon_decay=0.995, buffer_capacity=10000,
                 batch_size=32, target_update=10):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self._step_counter = 0

        # Build a state encoding that works well: one-hot or raw?
        # Use one-hot encoding of state (16-dim) → hidden → actions
        state_dim = n_states  # one-hot
        self.q_network = MLP([state_dim, hidden_dim, n_actions], lr=lr)
        self.target_network = MLP([state_dim, hidden_dim, n_actions], lr=lr)
        self.target_network.copy_from(self.q_network)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def _encode(self, state):
        """One-hot encode the state (list of floats)."""
        enc = [0.0] * len(self.q_network.weights[0][0])  # input dim
        if state < len(enc):
            enc[state] = 1.0
        return enc

    def act(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_enc = self._encode(state)
        q_vals = self.q_network.forward(state_enc)
        return _argmax(q_vals)

    def learn(self, state, action, reward, next_state, done):
        """Store transition and train if enough data."""
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        total_loss = 0.0

        for s, a, r, ns, d in batch:
            s_enc = self._encode(s)
            ns_enc = self._encode(ns)

            q_pred = self.q_network.forward(s_enc)

            if d:
                target_val = r
            else:
                next_q = self.target_network.forward(ns_enc)
                target_val = r + self.gamma * max(next_q)

            targets = q_pred[:]  # copy
            targets[a] = target_val

            # MSE backward
            self.q_network.backward_mse(targets, q_pred)
            total_loss += (q_pred[a] - target_val) ** 2

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self._step_counter += 1
        if self._step_counter % self.target_update == 0:
            self.target_network.copy_from(self.q_network)

        return total_loss / self.batch_size

    def greedy_policy(self):
        policy = [0] * 16
        for s in range(16):
            if s == 15:
                continue
            s_enc = self._encode(s)
            q_vals = self.q_network.forward(s_enc)
            policy[s] = _argmax(q_vals)
        return policy


# ═══════════════════════════════════════════════════════════════
#  PPO — Proximal Policy Optimization
# ═══════════════════════════════════════════════════════════════

class PPO:
    """PPO with clipped surrogate objective and GAE advantage estimation.

    Actor network: state → logits → softmax action probabilities.
    Critic network: state → scalar value.
    """

    def __init__(self, n_states=16, n_actions=4, hidden_dim=16,
                 actor_lr=0.005, critic_lr=0.01, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=4):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs

        state_dim = n_states  # one-hot

        self.actor = MLP([state_dim, hidden_dim, n_actions], lr=actor_lr)
        self.critic = MLP([state_dim, hidden_dim, 1], lr=critic_lr)

    def _encode(self, state):
        """One-hot state encoding."""
        enc = [0.0] * len(self.actor.weights[0][0])
        if state < len(enc):
            enc[state] = 1.0
        return enc

    def act(self, state):
        """Sample action from stochastic policy (no ε-greedy — PPO explores via its own distribution).

        Returns (action, log_prob_of_action).
        """
        s_enc = self._encode(state)
        logits = self.actor.forward(s_enc)
        logits = [max(-20.0, min(20.0, l)) for l in logits]
        probs = _softmax(logits)

        # Sample from categorical distribution
        r = random.random()
        cum = 0.0
        action = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                action = i
                break
        log_prob = math.log(max(probs[action], 1e-15))
        return action, log_prob

    def get_value(self, state):
        s_enc = self._encode(state)
        return self.critic.forward(s_enc)[0]

    def _compute_gae(self, rewards, values, dones):
        """Compute GAE advantages and returns.

        values has length T+1 (last is bootstrap from terminal state).
        Uses safe math to avoid overflow.
        """
        T = len(rewards)
        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            done_flag = 0.0 if dones[t] else 1.0
            delta = (rewards[t]
                     + self.gamma * values[t + 1] * done_flag
                     - values[t])
            # Clamp delta to avoid compounding extremes
            delta = max(-50.0, min(50.0, delta))
            values[t + 1] = max(-50.0, min(50.0, values[t + 1]))
            gae = delta + self.gamma * self.lam * done_flag * gae
            gae = max(-50.0, min(50.0, gae))
            advantages[t] = gae
        returns = [advantages[t] + values[t] for t in range(T)]
        return advantages, returns

    def train_episode(self, states, actions, old_log_probs, rewards, dones):
        """Train on a collected episode using PPO update."""
        # Bootstrap value
        if dones[-1]:
            bootstrap_val = 0.0
        else:
            raw = self.get_value(states[-1])
            bootstrap_val = max(-50.0, min(50.0, raw))

        values = []
        for s in states:
            raw_val = self.get_value(s)
            values.append(max(-50.0, min(50.0, raw_val)))
        values.append(bootstrap_val)

        advantages, returns = self._compute_gae(rewards, values, dones)

        # Normalize advantages with overflow-safe math
        n = len(advantages)
        if n > 0:
            mean_adv = sum(advantages) / n
            # Use mean absolute deviation when variance would overflow
            max_dev = max(abs(a - mean_adv) for a in advantages)
            if max_dev < 1e-30:
                advantages = [0.0] * n
            else:
                # Safe variance computation — cap each term
                sq_sum = 0.0
                for a in advantages:
                    d = a - mean_adv
                    if abs(d) > 1e10:
                        d = 1e10 if d > 0 else -1e10
                    sq_sum += d * d
                var_adv = sq_sum / n
                std_adv = math.sqrt(var_adv + 1e-12)
                # Clip normalized advantages to [-5, 5]
                advantages = [max(-5.0, min(5.0, (a - mean_adv) / std_adv))
                              for a in advantages]

        # Multi-epoch PPO update
        for _ in range(self.epochs):
            for i in range(len(states)):
                s = states[i]
                a = actions[i]
                old_lp = old_log_probs[i]
                adv = advantages[i]
                ret = returns[i]

                s_enc = self._encode(s)

                # ── Actor update ──
                logits = self.actor.forward(s_enc)
                # Guard against extreme logits
                logits = [max(-20.0, min(20.0, l)) for l in logits]
                probs = _softmax(logits)
                new_log_prob = math.log(max(probs[a], 1e-15))

                # Clamp ratio to avoid extreme updates
                ratio = math.exp(max(-10.0, min(10.0, new_log_prob - old_lp)))

                # Determine if clipped
                if adv >= 0:
                    not_clipped = ratio <= 1.0 + self.clip_eps
                else:
                    not_clipped = ratio >= 1.0 - self.clip_eps

                # Gradient w.r.t. logits of log_prob(a)
                grad_log_prob = [-p for p in probs]
                grad_log_prob[a] = 1.0 - probs[a]

                if not_clipped:
                    # dL/d(logits) = -adv * ratio * grad_log_prob
                    actor_grad = [-adv * ratio * g for g in grad_log_prob]
                    # Clip gradient to avoid exploding updates
                    actor_grad = [max(-10.0, min(10.0, g)) for g in actor_grad]
                else:
                    actor_grad = [0.0] * self.n_actions

                self.actor.backward_custom(actor_grad)

                # ── Critic update ──
                v_pred = self.critic.forward(s_enc)[0]
                # Clamp return as well for stable critic
                ret = max(-50.0, min(50.0, ret))
                # MSE gradient with clipping
                err = v_pred - ret
                err = max(-10.0, min(10.0, err))
                critic_grad = [2.0 * err]
                self.critic.backward_custom(critic_grad)

    def greedy_policy(self):
        """Extract deterministic policy from actor."""
        policy = [0] * 16
        for s in range(16):
            if s == 15:
                continue
            s_enc = self._encode(s)
            logits = self.actor.forward(s_enc)
            policy[s] = _argmax(logits)
        return policy


# ═══════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════

def train_qlearning(env, episodes=200, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Train Q-Learning and return episode returns + final agent."""
    agent = QLearning(env.N_STATES, env.N_ACTIONS, alpha, gamma, epsilon)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            a = agent.act(s)
            ns, r, done = env.step(s, a)
            agent.learn(s, a, r, ns, done)
            ep_return += r
            s = ns
        returns.append(ep_return)
    return agent, returns


def train_sarsa(env, episodes=200, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Train SARSA and return episode returns + final agent."""
    agent = SARSA(env.N_STATES, env.N_ACTIONS, alpha, gamma, epsilon)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        a = agent.act(s)
        ep_return = 0.0
        done = False
        while not done:
            ns, r, done = env.step(s, a)
            if not done:
                na = agent.act(ns)
            else:
                na = 0
            agent.learn(s, a, r, ns, na, done)
            ep_return += r
            s, a = ns, na
        returns.append(ep_return)
    return agent, returns


def train_dqn(env, episodes=300, **kwargs):
    """Train DQN and return episode returns + final agent."""
    agent = DQN(env.N_STATES, env.N_ACTIONS, **kwargs)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            a = agent.act(s)
            ns, r, done = env.step(s, a)
            agent.learn(s, a, r, ns, done)
            ep_return += r
            s = ns
        returns.append(ep_return)
    return agent, returns


def train_ppo(env, episodes=300, **kwargs):
    """Train PPO and return episode returns + final agent."""
    agent = PPO(env.N_STATES, env.N_ACTIONS, **kwargs)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        ep_return = 0.0
        done = False
        while not done:
            a, lp = agent.act(s)
            ns, r, done = env.step(s, a)
            states.append(s)
            actions.append(a)
            log_probs.append(lp)
            rewards.append(r)
            dones.append(done)
            ep_return += r
            s = ns
        # Train on the episode
        agent.train_episode(states, actions, log_probs, rewards, dones)
        returns.append(ep_return)
    return agent, returns


# ═══════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════

def demo():
    """Run all RL algorithms on GridWorld and show results."""
    print("=" * 64)
    print("  🤖  Anggira RL — Reinforcement Learning from Scratch")
    print("=" * 64)

    env = GridWorld()

    # ── 1. Dynamic Programming ──
    print("\n" + "─" * 64)
    print("  📐  DYNAMIC PROGRAMMING")
    print("─" * 64)

    print("\n  Policy Iteration — Optimal Policy:")
    pi_policy, pi_values = policy_iteration(env)
    print("     " + env.render_policy(pi_policy).replace("\n", "\n     "))
    print("\n  State Values (Policy Iteration):")
    print("     " + env.render_values(pi_values).replace("\n", "\n     "))

    print("\n  Value Iteration — Optimal Policy:")
    vi_policy, vi_values = value_iteration(env)
    print("     " + env.render_policy(vi_policy).replace("\n", "\n     "))

    # ── 2. Q-Learning ──
    print("\n" + "─" * 64)
    print("  🧠  Q-LEARNING  (Tabular, α=0.1, γ=0.9, ε=0.1)")
    print("─" * 64)
    q_agent, q_returns = train_qlearning(env, episodes=200)
    q_last50 = sum(q_returns[-50:]) / 50
    print(f"  Final 50-ep avg return: {q_last50:.1f}")
    print(f"  Returns plot (200 eps): {_text_plot(q_returns)}")
    print(f"  Greedy policy:")
    print("     " + env.render_policy(q_agent.greedy_policy()).replace("\n", "\n     "))

    # ── 3. SARSA ──
    print("\n" + "─" * 64)
    print("  🧠  SARSA  (Tabular, α=0.1, γ=0.9, ε=0.1)")
    print("─" * 64)
    s_agent, s_returns = train_sarsa(env, episodes=200)
    s_last50 = sum(s_returns[-50:]) / 50
    print(f"  Final 50-ep avg return: {s_last50:.1f}")
    print(f"  Returns plot (200 eps): {_text_plot(s_returns)}")
    print(f"  Greedy policy:")
    print("     " + env.render_policy(s_agent.greedy_policy()).replace("\n", "\n     "))

    # ── 4. DQN ──
    print("\n" + "─" * 64)
    print("  🤖  DQN  (Deep Q-Network, 16→16→4 MLP)")
    print("─" * 64)
    dqn_agent, dqn_returns = train_dqn(env, episodes=300,
                                       hidden_dim=16, lr=0.01,
                                       gamma=0.9, epsilon=0.5,
                                       epsilon_min=0.01, epsilon_decay=0.99,
                                       buffer_capacity=5000, batch_size=16,
                                       target_update=20)
    dqn_last50 = sum(dqn_returns[-50:]) / 50
    final_eps = dqn_agent.epsilon
    print(f"  Final ε: {final_eps:.3f}")
    print(f"  Final 50-ep avg return: {dqn_last50:.1f}")
    print(f"  Returns plot (300 eps): {_text_plot(dqn_returns)}")
    print(f"  Greedy policy:")
    print("     " + env.render_policy(dqn_agent.greedy_policy()).replace("\n", "\n     "))

    # ── 5. PPO ──
    print("\n" + "─" * 64)
    print("  🎭  PPO  (Clipped Surrogate + GAE, 16→16→4 actor)")
    print("─" * 64)
    ppo_agent, ppo_returns = train_ppo(env, episodes=400,
                                       hidden_dim=16,
                                       actor_lr=0.003, critic_lr=0.005,
                                       gamma=0.99, lam=0.95, clip_eps=0.2,
                                       epochs=4)
    ppo_last50 = sum(ppo_returns[-50:]) / 50
    print(f"  Final 50-ep avg return: {ppo_last50:.1f}")
    print(f"  Returns plot (400 eps): {_text_plot(ppo_returns)}")
    print(f"  Greedy policy:")
    print("     " + env.render_policy(ppo_agent.greedy_policy()).replace("\n", "\n     "))

    # ── Summary ──
    print("\n" + "=" * 64)
    print("  📊  SUMMARY — Final 50-episode Average Returns")
    print("=" * 64)
    print(f"    DP (optimal)      —  (exact via value iteration)")
    print(f"    Q-Learning         {q_last50:8.1f}")
    print(f"    SARSA              {s_last50:8.1f}")
    print(f"    DQN                {dqn_last50:8.1f}")
    print(f"    PPO                {ppo_last50:8.1f}")
    print()

    # Quick sanity: print optimal return from DP
    _, opt_vals = value_iteration(env)
    print(f"  Optimal V(0) = {opt_vals[0]:.2f} "
          f"(expected return from start under optimal policy)")
    print(f"  DP Theoretical best from state 0: about -9 to -13 steps × -1 reward")
    print()
    print("  ✅ All algorithms trained successfully. All pure Python — no numpy!")


if __name__ == "__main__":
    demo()
