"""
Anggira Probability — Distributions, Sampling, and Information Theory

Phase 1: Math Foundations (Lessons 6-7, 9)
"""

import math
import random

random.seed(42)


# ═══════════════════════════════════════════════
# COMBINATORICS
# ═══════════════════════════════════════════════

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


# ═══════════════════════════════════════════════
# CONDITIONAL PROBABILITY & BAYES
# ═══════════════════════════════════════════════

def conditional_probability(p_a_and_b, p_b):
    """P(A|B) = P(A∩B) / P(B)"""
    return p_a_and_b / p_b


def bayes_rule(p_b_given_a, p_a, p_b):
    """P(A|B) = P(B|A) * P(A) / P(B)"""
    return p_b_given_a * p_a / p_b


def bayes_theorem(prior, likelihood, evidence):
    """General Bayes: posterior ∝ likelihood × prior"""
    posterior = [likelihood[i] * prior[i] / evidence for i in range(len(prior))]
    return posterior


# ═══════════════════════════════════════════════
# PMFs
# ═══════════════════════════════════════════════

def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    return probs[k]


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)


# ═══════════════════════════════════════════════
# PDFs
# ═══════════════════════════════════════════════

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def normal_cdf(x, mu=0, sigma=1):
    """CDF via error function approximation."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


# ═══════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════

def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


# ═══════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════

def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples


def sample_uniform(a, b, n=1):
    return [a + (b - a) * random.random() for _ in range(n)]


def sample_normal_box_muller(mu, sigma, n=1):
    """Sample from normal distribution using Box-Muller transform."""
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples


# ═══════════════════════════════════════════════
# SOFTMAX & CROSS-ENTROPY
# ═══════════════════════════════════════════════

def softmax(logits):
    """Softmax with numerical stability (subtract max)."""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    """Log-softmax for numerical stability."""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    """Cross-entropy loss from logits (no softmax wrapper — stable)."""
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


# ═══════════════════════════════════════════════
# INFORMATION THEORY (Lesson 9)
# ═══════════════════════════════════════════════

def entropy(probs):
    """H(X) = -Σ p(x) log₂ p(x) — expected information content."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def cross_entropy(p_probs, q_probs):
    """H(p, q) = -Σ p(x) log₂ q(x)"""
    return -sum(p * math.log2(q) for p, q in zip(p_probs, q_probs) if p > 0)


def kl_divergence(p_probs, q_probs):
    """D_KL(p || q) = Σ p(x) log₂(p(x)/q(x))"""
    return sum(p * math.log2(p / q) for p, q in zip(p_probs, q_probs) if p > 0)


def mutual_information(joint, marginal_x, marginal_y):
    """I(X;Y) = ΣΣ p(x,y) log₂(p(x,y)/(p(x)p(y)))"""
    mi = 0.0
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if joint[i][j] > 0:
                mi += joint[i][j] * math.log2(
                    joint[i][j] / (marginal_x[i] * marginal_y[j])
                )
    return mi


# ═══════════════════════════════════════════════
# JOINT & MARGINAL
# ═══════════════════════════════════════════════

def joint_to_marginals(joint):
    rows = len(joint)
    cols = len(joint[0])
    marginal_x = [sum(joint[i][j] for j in range(cols)) for i in range(rows)]
    marginal_y = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]
    return marginal_x, marginal_y


def check_independence(joint, marginal_x, marginal_y, tol=1e-9):
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if abs(joint[i][j] - marginal_x[i] * marginal_y[j]) > tol:
                return False
    return True


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo():
    print("🤖 Anggira — Probability, Bayes & Information Theory")
    print("=" * 60)

    print("\n--- Bayes Theorem ---")
    prior = [0.5, 0.5]  # P(sick)=0.5, P(healthy)=0.5
    likelihood = [0.95, 0.10]  # P(pos|sick)=0.95, P(pos|healthy)=0.10
    evidence = prior[0] * likelihood[0] + prior[1] * likelihood[1]
    posterior = bayes_theorem(prior, likelihood, evidence)
    print(f"  P(sick|positive) = {posterior[0]:.4f}")
    print(f"  P(healthy|positive) = {posterior[1]:.4f}")

    print("\n--- Softmax ---")
    probs = softmax([2.0, 1.0, 0.1])
    print(f"  Softmax([2, 1, 0.1]): {[f'{p:.4f}' for p in probs]}")

    print("\n--- Information Theory ---")
    fair_coin = [0.5, 0.5]
    biased_coin = [0.9, 0.1]
    print(f"  H(fair coin) = {entropy(fair_coin):.4f} bits")
    print(f"  H(biased coin) = {entropy(biased_coin):.4f} bits")
    print(f"  D_KL(fair || biased) = {kl_divergence(fair_coin, biased_coin):.4f} bits")

    joint = [[0.4, 0.1], [0.05, 0.45]]
    mx, my = joint_to_marginals(joint)
    mi = mutual_information(joint, mx, my)
    print(f"  I(weather; umbrella) = {mi:.4f} bits")
    print(f"  Independent? {check_independence(joint, mx, my)}")

    print("\n--- Normal Distribution ---")
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        print(f"  N({x:+d} | 0, 1) = {normal_pdf(x, 0, 1):.4f}")

    print("\n--- Sampling ---")
    samples = sample_normal_box_muller(5, 2, 10000)
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    print(f"  N(5, 2) 10k samples: mean={mean:.3f}, var={var:.3f}")


if __name__ == "__main__":
    demo()
