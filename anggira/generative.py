"""Anggira Generative AI — VAE, GAN, DDPM, Flow Matching, LoRA

All pure Python (stdlib only) using list-of-lists matrices.
Synthetic 1D/2D data for training examples.

Components:
  - VAE:     Variational Autoencoder (reparameterization, KL loss)
  - GAN:     Generative Adversarial Network (adversarial training, mode collapse detection)
  - DDPM:    Denoising Diffusion Probabilistic Models (forward/reverse, UNet-lite)
  - Flow:    Flow Matching (straight-line flow, Euler integration)
  - LoRA:    Low-Rank Adaptation (A/B matrices applied to linear layer)
"""

import math
import random


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _relu(x):
    return max(0.0, x)


def _relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def _sigmoid(x):
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x


def _leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha


def _softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _mse(a, b):
    """Mean squared error between two lists."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def _mat_mul(A, B):
    """Matrix multiplication: A (m×n) * B (n×p) → (m×p)."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
            for i in range(m)]


def _mat_vec_mul(A, v):
    """Matrix (m×n) × vector (n) → vector (m)."""
    return [sum(A[i][k] * v[k] for k in range(len(v))) for i in range(len(A))]


def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _vec_scale(v, s):
    return [x * s for x in v]


def _outer(a, b):
    """Outer product of vectors a (m) and b (n) → matrix (m×n)."""
    return [[x * y for y in b] for x in a]


def _zeros(shape):
    """Return a zero matrix (rows × cols) or zero vector."""
    if isinstance(shape, int):
        return [0.0] * shape
    return [[0.0] * shape[1] for _ in range(shape[0])]


def _randn_vec(dim, scale=1.0):
    """Box-Muller normal random vector."""
    out = []
    for _ in range(dim):
        u1 = random.random() or 1e-10
        u2 = random.random() or 1e-10
        out.append(scale * math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2))
    return out


def _randn_matrix(rows, cols, scale=1.0):
    """Matrix of Gaussian random numbers."""
    return [_randn_vec(cols, scale) for _ in range(rows)]


def _make_linear(in_dim, out_dim, scale=None):
    """Create weight matrix and bias vector."""
    if scale is None:
        scale = math.sqrt(2.0 / in_dim)
    w = _randn_matrix(out_dim, in_dim, scale)
    b = [0.0] * out_dim
    return w, b


# ═══════════════════════════════════════════════════════════════
#  SIMPLE MLP LAYER  (forward + backward)
# ═══════════════════════════════════════════════════════════════

class LinearLayer:
    """A single linear layer with weights, bias, and gradient updates.

    w: (out_dim × in_dim) matrix (list of lists)
    b: bias vector (length out_dim)
    """

    def __init__(self, in_dim, out_dim, lr=0.01, activation='relu'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.activation = activation
        scale = math.sqrt(2.0 / in_dim)
        self.w = _randn_matrix(out_dim, in_dim, scale)
        self.b = [0.0] * out_dim

    def forward(self, x):
        """x is input vector. Returns pre-activation z and post-activation a."""
        z = _mat_vec_mul(self.w, x)
        z = _vec_add(z, self.b)
        if self.activation == 'relu':
            a = [_relu(v) for v in z]
        elif self.activation == 'leaky_relu':
            a = [_leaky_relu(v) for v in z]
        elif self.activation == 'linear':
            a = z[:]
        elif self.activation == 'sigmoid':
            a = [_sigmoid(v) for v in z]
        else:
            a = z[:]
        self._input = x
        self._z = z
        self._output = a
        return a

    def backward(self, grad_output):
        """grad_output: gradient w.r.t. output.
        Returns gradient w.r.t. input.
        Updates self.w and self.b.
        """
        x = self._input
        z = self._z

        # Gradient through activation
        if self.activation == 'relu':
            dz = [grad_output[j] * _relu_derivative(z[j]) for j in range(self.out_dim)]
        elif self.activation == 'leaky_relu':
            dz = [grad_output[j] * _leaky_relu_derivative(z[j]) for j in range(self.out_dim)]
        elif self.activation == 'sigmoid':
            s = [_sigmoid(v) for v in z]
            dz = [grad_output[j] * s[j] * (1 - s[j]) for j in range(self.out_dim)]
        else:  # linear
            dz = grad_output[:]

        # Update weights: w[j][k] -= lr * dz[j] * x[k]
        for j in range(self.out_dim):
            self.b[j] -= self.lr * dz[j]
            for k in range(self.in_dim):
                self.w[j][k] -= self.lr * dz[j] * x[k]

        # Gradient w.r.t. input: grad_in[k] = sum_j dz[j] * w[j][k]
        grad_in = [0.0] * self.in_dim
        for k in range(self.in_dim):
            for j in range(self.out_dim):
                grad_in[k] += dz[j] * self.w[j][k]
        return grad_in


# ═══════════════════════════════════════════════════════════════
#  1. VAE — Variational Autoencoder
# ═══════════════════════════════════════════════════════════════

class VAE:
    """Variational Autoencoder.

    Architecture:
      Encoder:  input_dim → hidden_dim → 2*latent_dim (mu + logvar)
      Decoder:  latent_dim → hidden_dim → input_dim

    Trained with reconstruction + KL divergence loss.
    Full reparameterization gradient flows through mu and logvar.
    """

    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=16, lr=0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Encoder: input → hidden (ReLU) → mu+logvar (linear)
        self.enc1 = LinearLayer(input_dim, hidden_dim, lr, activation='relu')
        self.enc_mu = LinearLayer(hidden_dim, latent_dim, lr, activation='linear')
        self.enc_logvar = LinearLayer(hidden_dim, latent_dim, lr, activation='linear')

        # Decoder: latent → hidden (ReLU) → input (sigmoid)
        self.dec1 = LinearLayer(latent_dim, hidden_dim, lr, activation='relu')
        self.dec_out = LinearLayer(hidden_dim, input_dim, lr, activation='sigmoid')

    def encode(self, x):
        """Return (mu, logvar)."""
        h = self.enc1.forward(x)
        mu = self.enc_mu.forward(h)
        logvar = self.enc_logvar.forward(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, exp(logvar)) using reparameterization trick.
        Stores std and eps for backward.
        """
        self._mu = mu
        self._logvar = logvar
        self._std = [math.exp(lv * 0.5) for lv in logvar]
        self._eps = _randn_vec(len(mu))
        return [mu[i] + self._std[i] * self._eps[i] for i in range(len(mu))]

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        h = self.dec1.forward(z)
        return self.dec_out.forward(h)

    def forward(self, x):
        """Full forward pass: encode → reparameterize → decode.
        Returns (reconstruction, mu, logvar, z).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def backward(self, recon, x, mu, logvar):
        """Backward pass with reconstruction + KL loss.
        Full chain: recon_loss → decoder → z → (mu, logvar) → encoder
        Returns (recon_loss, kl_loss).
        """
        eps = 1e-12

        # ── Reconstruction loss (BCE) ──
        recon_loss = -sum(
            x[i] * math.log(max(recon[i], eps)) + (1 - x[i]) * math.log(max(1 - recon[i], eps))
            for i in range(self.input_dim)
        )

        # Gradient of BCE w.r.t. decoder output
        grad_recon = []
        for i in range(self.input_dim):
            rc = max(recon[i], eps)
            grad_recon.append(-(x[i] / rc - (1.0 - x[i]) / max(1.0 - rc, eps)))

        # ── Backward through decoder → get gradient w.r.t. z (latent code) ──
        grad_hidden = self.dec_out.backward(grad_recon)
        grad_z = self.dec1.backward(grad_hidden)  # gradient w.r.t. z (input to dec1)

        # ── KL divergence ──
        kl_loss = -0.5 * sum(
            1.0 + logvar[i] - mu[i] ** 2 - math.exp(logvar[i])
            for i in range(self.latent_dim)
        )

        # dKL/dmu = mu,  dKL/dlogvar = 0.5 * (exp(logvar) - 1)
        grad_kl_mu = [mu[i] for i in range(self.latent_dim)]
        grad_kl_logvar = [0.5 * (math.exp(logvar[i]) - 1.0) for i in range(self.latent_dim)]

        # ── Reparameterization gradients ──
        # z = mu + std * eps
        # d_z/d_mu = 1
        # d_z/d_logvar = eps * 0.5 * std  (since std = exp(logvar/2))
        # So gradient w.r.t. mu = grad_z * 1 + grad_kl_mu
        # gradient w.r.t. logvar = grad_z * eps * 0.5 * std + grad_kl_logvar
        grad_mu = [grad_z[i] + grad_kl_mu[i] for i in range(self.latent_dim)]
        grad_logvar = [
            grad_z[i] * self._eps[i] * 0.5 * self._std[i] + grad_kl_logvar[i]
            for i in range(self.latent_dim)
        ]

        # ── Backward through encoder ──
        # Combine gradients from mu and logvar paths at the shared hidden layer
        grad_enc_hidden = [0.0] * self.hidden_dim
        for k in range(self.hidden_dim):
            for j in range(self.latent_dim):
                grad_enc_hidden[k] += grad_mu[j] * self.enc_mu.w[j][k]
            for j in range(self.latent_dim):
                grad_enc_hidden[k] += grad_logvar[j] * self.enc_logvar.w[j][k]

        self.enc_mu.backward(grad_mu)
        self.enc_logvar.backward(grad_logvar)
        self.enc1.backward(grad_enc_hidden)

        return recon_loss, kl_loss

    def train_step(self, x):
        """Single training step on input x. Returns (recon_loss, kl_loss)."""
        recon, mu, logvar, z = self.forward(x)
        return self.backward(recon, x, mu, logvar)

    def generate(self, z=None):
        """Generate a sample. If z is None, sample from N(0, I)."""
        if z is None:
            z = _randn_vec(self.latent_dim)
        return self.decode(z)


# ═══════════════════════════════════════════════════════════════
#  2. GAN — Generative Adversarial Network
# ═══════════════════════════════════════════════════════════════

class GAN:
    """Generative Adversarial Network.

    Generator:     noise_dim → hidden → data_dim (sigmoid)
    Discriminator: data_dim → hidden → 1 (sigmoid)

    Adversarial training with binary cross-entropy.
    Mode collapse detection via variance of generated samples.
    """

    def __init__(self, data_dim=2, noise_dim=4, hidden_dim=16, lr=0.01):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Generator
        self.gen1 = LinearLayer(noise_dim, hidden_dim, lr, activation='relu')
        self.gen_out = LinearLayer(hidden_dim, data_dim, lr, activation='sigmoid')

        # Discriminator
        self.disc1 = LinearLayer(data_dim, hidden_dim, lr, activation='leaky_relu')
        self.disc_out = LinearLayer(hidden_dim, 1, lr, activation='sigmoid')

    def generate(self, z=None):
        """Generate fake data from noise."""
        if z is None:
            z = _randn_vec(self.noise_dim)
        h = self.gen1.forward(z)
        return self.gen_out.forward(h)

    def discriminate(self, x):
        """Binary classification: is x real (1) or fake (0)?"""
        h = self.disc1.forward(x)
        return self.disc_out.forward(h)

    def train_discriminator(self, real_data, fake_data):
        """Train discriminator on real (label=1) and fake (label=0) data.
        Returns (d_real_loss, d_fake_loss).
        """
        eps = 1e-12

        # Real data: D(x) should → 1
        d_real = self.discriminate(real_data)
        d_real_loss = -math.log(max(d_real[0], eps))
        grad_real = [-(1.0 / max(d_real[0], eps))]
        grad_h = self.disc_out.backward(grad_real)
        self.disc1.backward(grad_h)

        # Fake data: D(G(z)) should → 0
        d_fake = self.discriminate(fake_data)
        d_fake_loss = -math.log(max(1.0 - d_fake[0], eps))
        grad_fake = [-(-1.0 / max(1.0 - d_fake[0], eps))]
        grad_h = self.disc_out.backward(grad_fake)
        self.disc1.backward(grad_h)

        return d_real_loss, d_fake_loss

    def train_generator(self, fake_data):
        """Train generator to fool discriminator: D(G(z)) → 1.
        Returns generator loss.
        """
        eps = 1e-12
        d_fake = self.discriminate(fake_data)
        g_loss = -math.log(max(d_fake[0], eps))

        # Gradient through discriminator (discriminator weights frozen for generator update)
        grad_d_fake = [-(1.0 / max(d_fake[0], eps))]

        # Backprop through discriminator output to get grad at fake_data
        # d(g_loss)/d(d_fake) = -1/d_fake
        # d(d_fake)/d(fake_data) = w_disc_out * d_sigmoid * w_disc1 * d_leaky_relu
        out_z = self.disc_out._z
        s = _sigmoid(out_z[0])
        ds = s * (1 - s)
        grad_at_out = grad_d_fake[0] * ds  # grad w.r.t. disc_out pre-activation

        # Backprop through disc_out weights: grad at disc1 output
        grad_at_disc1 = [grad_at_out * self.disc_out.w[0][j] for j in range(self.hidden_dim)]

        # Through disc1 activation (leaky_relu)
        for j in range(self.hidden_dim):
            grad_at_disc1[j] *= _leaky_relu_derivative(self.disc1._z[j])

        # Backprop through disc1 weights: grad at fake_data (generator output)
        grad_at_gen_out = [0.0] * self.data_dim
        for k in range(self.data_dim):
            for j in range(self.hidden_dim):
                grad_at_gen_out[k] += grad_at_disc1[j] * self.disc1.w[j][k]

        # Backward through generator
        grad_h = self.gen_out.backward(grad_at_gen_out)
        self.gen1.backward(grad_h)

        return g_loss

    def train_step(self, real_data):
        """Single GAN training step.
        Returns (d_loss, g_loss).
        """
        # Generate fake data
        z = _randn_vec(self.noise_dim)
        fake_data = self.generate(z)

        # Train discriminator first
        d_real_loss, d_fake_loss = self.train_discriminator(real_data, fake_data)
        d_loss = d_real_loss + d_fake_loss

        # Train generator (use fresh noise to avoid D getting too easy)
        z2 = _randn_vec(self.noise_dim)
        fake_data2 = self.generate_inference(z2)
        g_loss = self.train_generator(fake_data2)

        return d_loss, g_loss

    def generate_inference(self, z=None):
        """Generate without storing activations (for generator training)."""
        if z is None:
            z = _randn_vec(self.noise_dim)
        h = self.gen1.forward(z)
        return self.gen_out.forward(h)

    def detect_mode_collapse(self, n_samples=100):
        """Detect mode collapse by measuring variance of generated samples.
        Returns (variance, collapsed) where collapsed is True if variance < threshold.
        """
        samples = [self.generate() for _ in range(n_samples)]
        # Compute per-dimension variance, then mean
        vars_per_dim = []
        for d in range(self.data_dim):
            mean_d = sum(s[d] for s in samples) / n_samples
            var_d = sum((s[d] - mean_d) ** 2 for s in samples) / n_samples
            vars_per_dim.append(var_d)
        avg_variance = sum(vars_per_dim) / self.data_dim
        collapsed = avg_variance < 0.01
        return avg_variance, collapsed


# ═══════════════════════════════════════════════════════════════
#  3. DDPM — Denoising Diffusion Probabilistic Models
# ═══════════════════════════════════════════════════════════════

class DDPM:
    """Denoising Diffusion Probabilistic Models (DDPM).

    Forward:  q(x_t | x_{t-1}) = N(sqrt(1-beta_t) * x_{t-1}, beta_t * I)
    Reverse:  p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 * I)

    Uses a UNet-lite denoiser (simple MLP with noise level conditioning).
    """

    def __init__(self, data_dim=2, hidden_dim=32, timesteps=100, lr=0.01):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.lr = lr

        # Cosine beta schedule (more robust than linear)
        self.betas = self._cosine_beta_schedule(timesteps)
        self.alphas = [1.0 - b for b in self.betas]
        self.alpha_bars = [1.0]
        for a in self.alphas:
            self.alpha_bars.append(self.alpha_bars[-1] * a)
        self.alpha_bars = self.alpha_bars[1:]

        # Denoiser: UNet-lite
        # Input: data_dim (noisy x) + 1 (t embedding) → hidden → hidden → data_dim
        self.net1 = LinearLayer(data_dim + 1, hidden_dim, lr, activation='relu')
        self.net2 = LinearLayer(hidden_dim, hidden_dim, lr, activation='relu')
        self.net_out = LinearLayer(hidden_dim, data_dim, lr, activation='linear')

    def _cosine_beta_schedule(self, T, s=0.008):
        """Cosine beta schedule as in 'Improved DDPM'."""
        steps = T + 1
        t = [i / T for i in range(steps)]
        alphas_bar = [math.cos((t_i + s) / (1 + s) * math.pi / 2.0) ** 2 for t_i in t]
        betas = []
        for i in range(1, steps):
            betas.append(min(1.0 - alphas_bar[i] / alphas_bar[i - 1], 0.999))
        return betas

    def _denoiser(self, x_t, t_normalized):
        """Forward pass of UNet-lite denoiser.
        x_t: noisy data vector
        t_normalized: time step normalized to [0, 1] (float)
        Returns predicted noise.
        """
        inp = x_t + [t_normalized]
        h = self.net1.forward(inp)
        h = self.net2.forward(h)
        return self.net_out.forward(h)

    def forward_process(self, x_0, t):
        """Add noise to x_0 at timestep t.
        Returns (x_t, noise) where noise is the pure Gaussian added.
        """
        noise = _randn_vec(self.data_dim)
        alpha_bar_t = self.alpha_bars[t]
        sqrt_ab = math.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = math.sqrt(1.0 - alpha_bar_t)
        x_t = [sqrt_ab * x_0[i] + sqrt_one_minus_ab * noise[i] for i in range(self.data_dim)]
        return x_t, noise

    def train_step(self, x_0):
        """Single training step on clean data x_0.
        Returns loss.
        """
        # Sample random timestep
        t = random.randint(0, self.timesteps - 1)
        t_norm = t / self.timesteps

        # Forward process: add noise
        x_t, noise = self.forward_process(x_0, t)

        # Predict noise
        noise_pred = self._denoiser(x_t, t_norm)

        # MSE loss
        loss = _mse(noise_pred, noise)

        # Backward
        grad = [2.0 * (noise_pred[i] - noise[i]) / self.data_dim for i in range(self.data_dim)]
        grad_h = self.net_out.backward(grad)
        grad_h = self.net2.backward(grad_h)
        self.net1.backward(grad_h)

        return loss

    def sample(self, n_steps=None):
        """Generate a sample via reverse diffusion.
        Returns the generated sample.
        """
        if n_steps is None:
            n_steps = self.timesteps

        # Start from pure noise
        x = _randn_vec(self.data_dim)

        for t in range(n_steps - 1, -1, -1):
            t_norm = t / self.timesteps

            # Predict noise
            noise_pred = self._denoiser(x, t_norm)

            # Compute x_{t-1}
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # mu = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * noise_pred)
            coef1 = 1.0 / math.sqrt(alpha_t)
            coef2 = beta_t / math.sqrt(max(1.0 - alpha_bar_t, 1e-12))

            mu = [coef1 * (x[i] - coef2 * noise_pred[i]) for i in range(self.data_dim)]

            if t > 0:
                sigma = math.sqrt(beta_t)
                z = _randn_vec(self.data_dim)
                x = [mu[i] + sigma * z[i] for i in range(self.data_dim)]
            else:
                x = mu

        return x


# ═══════════════════════════════════════════════════════════════
#  4. FLOW MATCHING  — Straight-line flow from noise to data
# ═══════════════════════════════════════════════════════════════

class FlowMatching:
    """Flow Matching with Conditional Flow Matching (CFM).

    Defines a probability path from noise (p_0 = N(0, I)) to data (p_1 ≈ data).
    The flow is a linear interpolation: x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ N(0, I) and x_1 ~ data.

    The vector field v(x_t, t) predicts the velocity dx/dt = x_1 - x_0.
    Trained with conditional flow matching loss.
    Generation via Euler integration of the ODE.
    """

    def __init__(self, data_dim=2, hidden_dim=32, lr=0.01):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Vector field network: v_theta(x_t, t) → velocity
        # Input: data_dim + 1 (t) → hidden → hidden → data_dim
        self.vf1 = LinearLayer(data_dim + 1, hidden_dim, lr, activation='relu')
        self.vf2 = LinearLayer(hidden_dim, hidden_dim, lr, activation='relu')
        self.vf_out = LinearLayer(hidden_dim, data_dim, lr, activation='linear')

    def _velocity_net(self, x_t, t):
        """Compute predicted velocity v_theta(x_t, t).
        x_t: position at time t
        t: time in [0, 1] (float)
        """
        inp = x_t + [t]
        h = self.vf1.forward(inp)
        h = self.vf2.forward(h)
        return self.vf_out.forward(h)

    def train_step(self, x_1):
        """Single training step.
        x_1: data sample (from the target distribution).

        Returns loss.
        """
        # Sample time uniformly
        t = random.random()

        # Sample noise
        x_0 = _randn_vec(self.data_dim)

        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        x_t = [(1.0 - t) * x_0[i] + t * x_1[i] for i in range(self.data_dim)]

        # Target velocity: v = x_1 - x_0
        v_target = [x_1[i] - x_0[i] for i in range(self.data_dim)]

        # Predicted velocity
        v_pred = self._velocity_net(x_t, t)

        # MSE loss
        loss = _mse(v_pred, v_target)

        # Backward pass
        grad = [2.0 * (v_pred[i] - v_target[i]) / self.data_dim for i in range(self.data_dim)]
        grad_h = self.vf_out.backward(grad)
        grad_h = self.vf2.backward(grad_h)
        self.vf1.backward(grad_h)

        return loss

    def sample(self, n_steps=50):
        """Generate a sample via Euler integration of the ODE.
        Integrates from t=0 to t=1 starting from pure noise.

        Returns the generated sample.
        """
        dt = 1.0 / n_steps

        # Start from pure noise
        x = _randn_vec(self.data_dim)

        # Euler integration
        for step in range(n_steps):
            t = step * dt + dt / 2.0  # midpoint

            # Predict velocity at current position
            v = self._velocity_net(x, t)

            # Euler step: x_{t+dt} = x_t + v * dt
            x = [x[i] + v[i] * dt for i in range(self.data_dim)]

        return x


# ═══════════════════════════════════════════════════════════════
#  5. LoRA — Low-Rank Adaptation
# ═══════════════════════════════════════════════════════════════

class LoRALayer:
    """Low-Rank Adaptation (LoRA) applied to a linear layer.

    Instead of fine-tuning W, we add a low-rank update:
        h = (W + A @ B) * x + b
    where A (out_dim × rank) and B (rank × in_dim) are the LoRA matrices.

    During adaptation, only A and B are updated; W stays frozen.
    """

    def __init__(self, w, b, rank=4, lr=0.01):
        """Wrap an existing (weight, bias) pair with LoRA.

        Args:
            w: original weight matrix (out_dim × in_dim) — frozen
            b: original bias vector — frozen
            rank: low-rank dimension
            lr: learning rate for A and B
        """
        self.out_dim = len(w)
        self.in_dim = len(w[0])
        self.rank = rank
        self.lr = lr

        # Frozen original weights
        self.w = [row[:] for row in w]
        self.b = b[:]

        # LoRA matrices — initialized so A*B ≈ 0 (A Gaussian small, B = 0)
        init_scale = 0.02
        self.A = _randn_matrix(self.out_dim, rank, init_scale)
        self.B = [[0.0] * self.in_dim for _ in range(rank)]

    def forward(self, x, use_lora=True):
        """Forward pass. If use_lora=False, behaves like original layer."""
        # Wx + b
        z = _mat_vec_mul(self.w, x)
        z = _vec_add(z, self.b)

        if use_lora:
            # A @ B @ x
            bx = _mat_vec_mul(self.B, x)        # (rank) vector
            lora_out = _mat_vec_mul(self.A, bx)  # (out_dim) vector
            z = _vec_add(z, lora_out)

        self._input = x
        self._z = z
        self._output = z
        # Also store Bx for backward
        if use_lora:
            self._bx = bx
        else:
            self._bx = [0.0] * self.rank
        return z

    def backward(self, grad_output):
        """Backward pass, updating only A and B.
        Returns gradient w.r.t. input (for downstream layers).
        """
        x = self._input
        grad = grad_output[:]

        # Gradient w.r.t. input (through W only, since W is frozen)
        grad_in = [0.0] * self.in_dim
        for k in range(self.in_dim):
            for j in range(self.out_dim):
                grad_in[k] += grad[j] * self.w[j][k]

        # Update A: dA[j][r] = grad[j] * (Bx)[r]
        for j in range(self.out_dim):
            for r in range(self.rank):
                self.A[j][r] -= self.lr * grad[j] * self._bx[r]

        # Update B: dB[r][k] = sum_j(grad[j] * A[j][r]) * x[k]
        for r in range(self.rank):
            sum_grad_a = sum(grad[j] * self.A[j][r] for j in range(self.out_dim))
            for k in range(self.in_dim):
                self.B[r][k] -= self.lr * sum_grad_a * x[k]

        return grad_in

    def get_adapted_weight(self):
        """Return the effective weight: W + A @ B (for inspection)."""
        ab = _mat_mul(self.A, self.B)  # (out_dim × in_dim)
        adapted = [[self.w[j][k] + ab[j][k] for k in range(self.in_dim)]
                   for j in range(self.out_dim)]
        return adapted


# ═══════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATORS
# ═══════════════════════════════════════════════════════════════

def _make_2d_moons(n=200, noise=0.05):
    """Synthetic 2D moons dataset (two interleaving crescents)."""
    points = []
    for i in range(n):
        angle = math.pi * random.random()
        if i < n // 2:
            x = math.cos(angle)
            y = math.sin(angle)
        else:
            x = 1.0 - math.cos(angle)
            y = 0.5 - math.sin(angle)
        x += random.gauss(0, noise)
        y += random.gauss(0, noise)
        points.append([x, y])
    return points


def _make_2d_gaussian_mixture(n=200, centers=4):
    """Synthetic 2D Gaussian mixture."""
    points = []
    for i in range(n):
        ci = i % centers
        angle = 2 * math.pi * ci / centers
        cx = 0.5 * math.cos(angle)
        cy = 0.5 * math.sin(angle)
        x = cx + random.gauss(0, 0.1)
        y = cy + random.gauss(0, 0.1)
        points.append([x, y])
    return points


def _make_1d_data(n=100):
    """Simple 1D synthetic data: mixture of two Gaussians."""
    data = []
    for i in range(n):
        if random.random() < 0.5:
            data.append([random.gauss(-1.0, 0.3)])
        else:
            data.append([random.gauss(1.0, 0.3)])
    return data


def _normalize_data(data):
    """Normalize data to [0, 1] range for sigmoid-output models (VAE, GAN)."""
    dim = len(data[0])
    mins = [min(d[i] for d in data) for i in range(dim)]
    maxs = [max(d[i] for d in data) for i in range(dim)]
    ranges = [maxs[i] - mins[i] if maxs[i] - mins[i] > 1e-12 else 1.0 for i in range(dim)]
    normed = [[(d[i] - mins[i]) / ranges[i] for i in range(dim)] for d in data]
    return normed, mins, maxs


def _denormalize(data, mins, maxs):
    """Denormalize back to original range."""
    ranges = [maxs[i] - mins[i] if maxs[i] - mins[i] > 1e-12 else 1.0 for i in range(len(mins))]
    return [[d[i] * ranges[i] + mins[i] for i in range(len(mins))] for d in data]


# ═══════════════════════════════════════════════════════════════
#  TEXT-BASED PLOT HELPER
# ═══════════════════════════════════════════════════════════════

def _text_scatter(points, width=30, height=15):
    """2D text scatter plot. points is list of (x, y) in [0, 1]."""
    if not points:
        return "(empty)"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    rx = max(mx_x - mn_x, 1e-12)
    ry = max(mx_y - mn_y, 1e-12)

    grid = [[' ' for _ in range(width)] for _ in range(height)]
    for p in points:
        col = int((p[0] - mn_x) / rx * (width - 1))
        row = int((1.0 - (p[1] - mn_y) / ry) * (height - 1))
        col = min(max(col, 0), width - 1)
        row = min(max(row, 0), height - 1)
        grid[row][col] = '•'

    lines = []
    for row in grid:
        lines.append(''.join(row))
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  DEMO  — Run all components with training examples
# ═══════════════════════════════════════════════════════════════

def demo():
    """Demonstrate all generative AI components."""
    print("=" * 60)
    print("ANGGIRA GENERATIVE AI — Demo")
    print("Pure Python (stdlib), no external dependencies")
    print("=" * 60)

    # ── Synthetic 2D data ──
    print("\n▶ Creating synthetic 2D data...")
    raw_data = _make_2d_moons(200)
    data_01, mins, maxs = _normalize_data(raw_data)
    print(f"   Generated {len(data_01)} points (2D moons)")
    print(f"   Normalized  →  [{mins[0]:.2f}, {maxs[0]:.2f}] × [{mins[1]:.2f}, {maxs[1]:.2f}]")

    # ═══════════════════════════════════════════════════════════
    #  1. VAE DEMO
    # ═══════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("1. VAE — Variational Autoencoder")
    print("─" * 60)

    vae = VAE(input_dim=2, latent_dim=2, hidden_dim=16, lr=0.01)
    print(f"   Architecture: {vae.input_dim}→16→{vae.latent_dim} (latent)→16→{vae.input_dim}")

    vae_losses = []
    n_epochs = 500
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        for pt in data_01:
            rl, kl = vae.train_step(pt)
            total_loss += rl + kl
            total_recon += rl
            total_kl += kl
        avg_loss = total_loss / len(data_01)
        vae_losses.append(avg_loss)
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch + 1:4d}: loss={avg_loss:.4f}  "
                  f"(recon={total_recon / len(data_01):.4f}, kl={total_kl / len(data_01):.4f})")

    # Generate samples from VAE
    print("\n   Generated samples (VAE):")
    vae_samples = []
    for _ in range(10):
        z = _randn_vec(vae.latent_dim)
        sample = vae.generate(z)
        vae_samples.append(sample[:])
    for i, s in enumerate(vae_samples[:6]):
        print(f"      {i}: [{s[0]:.3f}, {s[1]:.3f}]")

    # ═══════════════════════════════════════════════════════════
    #  2. GAN DEMO
    # ═══════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("2. GAN — Generative Adversarial Network")
    print("─" * 60)

    gan = GAN(data_dim=2, noise_dim=4, hidden_dim=16, lr=0.005)
    print(f"   Generator: {gan.noise_dim}→16→{gan.data_dim} (sigmoid)")
    print(f"   Discriminator: {gan.data_dim}→16→1 (sigmoid)")

    gan_d_losses = []
    gan_g_losses = []
    n_steps = 1000
    for step in range(n_steps):
        real_pt = random.choice(data_01)
        # Pack as a batch of 1 (call train_step per-sample for simplicity)
        d_loss, g_loss = gan.train_step(real_pt)
        gan_d_losses.append(d_loss)
        gan_g_losses.append(g_loss)

        if (step + 1) % 200 == 0:
            if (step + 1) % 200 == 0:
                var, collapsed = gan.detect_mode_collapse(50)
                status = "⚠ COLLAPSED" if collapsed else "OK"
                print(f"   Step {step + 1:4d}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}  "
                      f"var={var:.4f} [{status}]")

    # Final mode collapse check
    final_var, final_collapsed = gan.detect_mode_collapse(100)
    print(f"\n   Final mode collapse variance: {final_var:.6f}  "
          f"({'⚠ COLLAPSED' if final_collapsed else '✓ OK'})")

    # Generated samples
    print("   Generated samples (GAN):")
    gan_samples = [gan.generate() for _ in range(10)]
    for i, s in enumerate(gan_samples[:6]):
        print(f"      {i}: [{s[0]:.3f}, {s[1]:.3f}]")

    # ═══════════════════════════════════════════════════════════
    #  3. DDPM DEMO
    # ═══════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("3. DDPM — Denoising Diffusion Probabilistic Models")
    print("─" * 60)

    ddpm = DDPM(data_dim=2, hidden_dim=32, timesteps=50, lr=0.01)
    print(f"   Denoiser: Input(data_dim+1)→32→32→data_dim")
    print(f"   Timesteps: {ddpm.timesteps}, Beta range: [{ddpm.betas[0]:.6f}, {ddpm.betas[-1]:.6f}]")

    # Train on normalized data
    ddpm_losses = []
    n_steps_ddpm = 500
    for step in range(n_steps_ddpm):
        pt = random.choice(data_01)
        loss = ddpm.train_step(pt)
        ddpm_losses.append(loss)
        if (step + 1) % 100 == 0:
            print(f"   Step {step + 1:4d}: loss={loss:.4f}")

    # Generate samples
    print("\n   Generated samples (DDPM):")
    ddpm_samples = [ddpm.sample() for _ in range(10)]
    for i, s in enumerate(ddpm_samples[:6]):
        print(f"      {i}: [{s[0]:.3f}, {s[1]:.3f}]")

    # ═══════════════════════════════════════════════════════════
    #  4. FLOW MATCHING DEMO
    # ═══════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("4. Flow Matching — Straight-Line Flow")
    print("─" * 60)

    # Use raw (non-normalized) data for flow matching
    fm_data = _make_2d_gaussian_mixture(200, centers=4)
    print(f"   Data: {len(fm_data)} points, 4 centers (Gaussian mixture)")

    fm = FlowMatching(data_dim=2, hidden_dim=32, lr=0.01)
    print(f"   Vector field net: v((x, t)) → ℝ²,  hidden=32")

    fm_losses = []
    n_steps_fm = 800
    for step in range(n_steps_fm):
        pt = random.choice(fm_data)
        loss = fm.train_step(pt)
        fm_losses.append(loss)
        if (step + 1) % 200 == 0:
            print(f"   Step {step + 1:4d}: loss={loss:.5f}")

    # Generate samples
    print("\n   Generated samples (Flow Matching):")
    fm_samples = [fm.sample(n_steps=30) for _ in range(10)]
    for i, s in enumerate(fm_samples[:6]):
        print(f"      {i}: [{s[0]:.3f}, {s[1]:.3f}]")

    # ═══════════════════════════════════════════════════════════
    #  5. LoRA DEMO
    # ═══════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("5. LoRA — Low-Rank Adaptation")
    print("─" * 60)

    # Create a base linear layer (e.g., a simple regression)
    in_dim = 4
    out_dim = 2
    w0, b0 = _make_linear(in_dim, out_dim)
    print(f"   Base layer: {in_dim}→{out_dim}, rank=2")

    def base_forward(w, b, x):
        return _vec_add(_mat_vec_mul(w, x), b)

    # Create synthetic task: learn a simple mapping
    # The original layer does something random; LoRA adapts it to a target function
    target_w = [[1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, -0.5]]
    target_b = [0.1, -0.1]

    # Create LoRA layer wrapping the pre-trained weights
    lora = LoRALayer(w0, b0, rank=2, lr=0.01)

    # Generate training data
    lora_data_x = [[random.gauss(0, 1) for _ in range(in_dim)] for _ in range(200)]
    lora_data_y = [base_forward(target_w, target_b, x) for x in lora_data_x]

    # Pre-LoRA error
    pre_lora_errors = []
    for x, y in zip(lora_data_x[:20], lora_data_y[:20]):
        pred = base_forward(w0, b0, x)
        err = sum((pred[i] - y[i]) ** 2 for i in range(out_dim))
        pre_lora_errors.append(err)
    print(f"   Pre-LoRA MSE (first 20): {sum(pre_lora_errors) / len(pre_lora_errors):.4f}")

    # Train LoRA
    lora_losses = []
    n_steps_lora = 500
    for step in range(n_steps_lora):
        idx = random.randrange(len(lora_data_x))
        x = lora_data_x[idx]
        y = lora_data_y[idx]

        # Forward with LoRA
        pred = lora.forward(x, use_lora=True)

        # MSE loss
        loss = sum((pred[i] - y[i]) ** 2 for i in range(out_dim))
        lora_losses.append(loss)

        # Backward
        grad = [2.0 * (pred[i] - y[i]) for i in range(out_dim)]
        lora.backward(grad)

        if (step + 1) % 100 == 0:
            print(f"   Step {step + 1:4d}: loss={loss:.6f}")

    # Post-LoRA error
    post_lora_errors = []
    for x, y in zip(lora_data_x[:20], lora_data_y[:20]):
        pred = lora.forward(x, use_lora=True)
        err = sum((pred[i] - y[i]) ** 2 for i in range(out_dim))
        post_lora_errors.append(err)
    print(f"   Post-LoRA MSE (first 20): {sum(post_lora_errors) / len(post_lora_errors):.6f}")

    # Show LoRA matrix statistics
    adapted_w = lora.get_adapted_weight()
    diff = [[adapted_w[j][k] - w0[j][k] for k in range(in_dim)] for j in range(out_dim)]
    print(f"\n   LoRA update magnitude: "
          f"ΔW max={max(max(abs(diff[j][k]) for k in range(in_dim)) for j in range(out_dim)):.4f}, "
          f"A norm={sum(sum(a**2 for a in row) for row in lora.A) ** 0.5:.4f}, "
          f"B norm={sum(sum(b**2 for b in row) for row in lora.B) ** 0.5:.4f}")

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("DEMO COMPLETE — Summary")
    print("=" * 60)
    print(f"  • VAE:            300 training epochs, final loss={vae_losses[-1]:.4f}")
    print(f"  • GAN:            1000 steps, mode collapse var={final_var:.6f}")
    print(f"  • DDPM:           500 steps, final loss={ddpm_losses[-1]:.4f}")
    print(f"  • Flow Matching:  800 steps, final loss={fm_losses[-1]:.4f}")
    print(f"  • LoRA:           500 steps, Δw max={max(max(abs(diff[j][k]) for k in range(in_dim)) for j in range(out_dim)):.4f}")
    print("=" * 60)


if __name__ == '__main__':
    demo()
