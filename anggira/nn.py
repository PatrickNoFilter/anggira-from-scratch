"""
Anggira Neural Network — Deep Learning from Scratch

Phase 3: Deep Learning Core
- Activation Functions, Optimizers (SGD, Adam, AdamW)
- Multi-Layer Perceptron, Training Loop
"""

import math
import random


# ═══════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════

def sigmoid(x):
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return math.tanh(x)


def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2


def relu(x):
    return max(0.0, x)


def relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x


def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha


def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ═══════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════

def mse_loss(y_pred, y_true):
    """Mean squared error."""
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_pred)


def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy with softmax built in (stable)."""
    logits = y_pred  # assume raw logits
    max_l = max(logits)
    log_sum_exp = max_l + math.log(sum(math.exp(z - max_l) for z in logits))
    return -sum(yt * (logit - log_sum_exp) for logit, yt in zip(logits, y_true))


# ═══════════════════════════════════════════════
# OPTIMIZERS
# ═══════════════════════════════════════════════

class SGD:
    """Stochastic Gradient Descent."""

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.params = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0] * len(parameters)

    def step(self):
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + p.grad
            p.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class Adam:
    """Adam optimizer with bias correction."""

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class AdamW:
    """Adam with decoupled weight decay."""

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            # Weight decay (decoupled)
            p.data -= self.lr * self.wd * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


# ═══════════════════════════════════════════════
# PARAMETER
# ═══════════════════════════════════════════════

class Param:
    """A trainable parameter."""

    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0


# ═══════════════════════════════════════════════
# LAYERS
# ═══════════════════════════════════════════════

class Dense:
    """A fully-connected (dense) neural network layer."""

    def __init__(self, n_inputs, n_outputs, activation='relu'):
        self.activation = activation
        # Xavier/Glorot initialization
        scale = math.sqrt(2.0 / (n_inputs + n_outputs)) if activation == 'relu' else math.sqrt(1.0 / n_inputs)
        self.w = [[Param(random.gauss(0, scale)) for _ in range(n_inputs)]
                  for _ in range(n_outputs)]
        self.b = [Param(0.0) for _ in range(n_outputs)]
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def forward(self, inputs):
        """Forward pass. inputs is a list of floats."""
        self._inputs = inputs
        self._z = []
        self._outputs = []
        for j in range(self.n_outputs):
            z = sum(self.w[j][k].data * inputs[k] for k in range(self.n_inputs)) + self.b[j].data
            self._z.append(z)
            if self.activation == 'relu':
                out = relu(z)
            elif self.activation == 'tanh':
                out = tanh(z)
            elif self.activation == 'sigmoid':
                out = sigmoid(z)
            elif self.activation == 'linear':
                out = z
            else:
                out = relu(z)
            self._outputs.append(out)
        return self._outputs

    def backward(self, grad_output):
        """Backward pass. grad_output is list of gradients for each output."""
        grad_input = [0.0] * self.n_inputs
        for j in range(self.n_outputs):
            if self.activation == 'relu':
                dz = relu_derivative(self._z[j])
            elif self.activation == 'tanh':
                dz = tanh_derivative(self._z[j])
            elif self.activation == 'sigmoid':
                dz = sigmoid_derivative(self._z[j])
            else:
                dz = 1.0

            dj = grad_output[j] * dz
            for k in range(self.n_inputs):
                self.w[j][k].grad += dj * self._inputs[k]
                grad_input[k] += dj * self.w[j][k].data
            self.b[j].grad += dj
        return grad_input

    def parameters(self):
        params = []
        for j in range(self.n_outputs):
            for k in range(self.n_inputs):
                params.append(self.w[j][k])
            params.append(self.b[j])
        return params


class NeuralNetwork:
    """Sequential neural network."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════

def train(model, X, y, optimizer, loss_fn='mse', epochs=100, verbose=True):
    """Train a neural network."""
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            y_pred = model.forward(xi)

            if loss_fn == 'mse':
                loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, yi)) / len(yi)
                grad = [2 * (yp - yt) / len(yi) for yp, yt in zip(y_pred, yi)]
            elif loss_fn == 'cross_entropy':
                probs = softmax(y_pred)
                loss = -math.log(max(probs[yi], 1e-15))
                grad = [probs[i] - (1.0 if i == yi else 0.0) for i in range(len(probs))]
            else:
                raise ValueError(f"Unknown loss: {loss_fn}")

            total_loss += loss
            optimizer.zero_grad()
            model.backward(grad)
            optimizer.step()

        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch:4d}: loss = {avg_loss:.6f}")
    return losses


def predict(model, X):
    """Make predictions (returns class index for classification)."""
    results = []
    for x in X:
        out = model.forward(x)
        results.append(out.index(max(out)))
    return results


def accuracy(model, X, y):
    preds = predict(model, X)
    return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo_xor():
    """Train Anggira on XOR using our scratch neural network."""
    print("🤖 Anggira — Neural Network Demo (XOR)")
    print("=" * 60)

    # XOR dataset
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]  # one-hot as class indices

    # Build network: 2 -> 4 -> 1
    model = NeuralNetwork([
        Dense(2, 4, activation='relu'),
        Dense(4, 2, activation='linear'),
    ])

    optimizer = Adam(model.parameters(), lr=0.01)
    y_onehot = [[1, 0], [0, 1], [0, 1], [1, 0]]

    print("\n  Training...")
    train(model, X, y_onehot, optimizer, loss_fn='mse', epochs=200)

    print("\n  Predictions:")
    for x in X:
        out = model.forward(x)
        pred = 0 if out[0] > out[1] else 1
        expected = y[X.index(x)]
        print(f"    {x} → {out} → {pred} (expected {expected}) {'✓' if pred == expected else '✗'}")


def demo_moons():
    """Train on synthetic moons dataset."""
    print("\n🤖 Anggira — Neural Network Demo (Moons)")
    print("=" * 60)
    random.seed(42)

    # Generate 2-ring moons dataset
    n = 200
    X, y = [], []
    for i in range(n):
        angle = random.uniform(0, 2 * math.pi)
        if i < n // 2:
            r = random.uniform(0.5, 1.0)
            label = 0
        else:
            r = random.uniform(1.5, 2.0)
            label = 1
        X.append([r * math.cos(angle) + random.gauss(0, 0.05),
                  r * math.sin(angle) + random.gauss(0, 0.05)])
        y.append(label)

    model = NeuralNetwork([
        Dense(2, 8, activation='relu'),
        Dense(8, 2, activation='linear'),
    ])
    optimizer = Adam(model.parameters(), lr=0.005)
    y_onehot = [[1, 0] if yi == 0 else [0, 1] for yi in y]

    print(f"\n  Dataset: {n} points, 2 classes")
    print("  Training...")
    train(model, X, y_onehot, optimizer, loss_fn='mse', epochs=100)

    acc = accuracy(model, X, y)
    print(f"\n  Final accuracy: {acc:.1%}")


if __name__ == "__main__":
    demo_xor()
    demo_moons()
