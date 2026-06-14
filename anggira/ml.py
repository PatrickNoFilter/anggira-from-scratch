"""
Anggira ML — Machine Learning Algorithms from Scratch

Phase 2: ML Fundamentals
- Linear/Logistic/Ridge Regression
- Decision Trees, KNN, Naive Bayes
"""

import math
import random


# ═══════════════════════════════════════════════
# LINEAR REGRESSION
# ═══════════════════════════════════════════════

class LinearRegression:
    """Linear regression using gradient descent."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n = len(X)
        # Support both list-of-list and list-of-scalar
        self._is_1d = not isinstance(X[0], (list, tuple))
        d = len(X[0]) if not self._is_1d else 1
        self.w = [0.0] * d
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = [self._predict(x) for x in X]
            dw = [0.0] * d
            db = 0.0
            for i in range(n):
                error = y_pred[i] - y[i]
                if self._is_1d:
                    dw[0] += error * X[i]
                    db += error
                else:
                    for j in range(d):
                        dw[j] += error * X[i][j]
                    db += error
            for j in range(d):
                self.w[j] -= self.lr * (dw[j] / n)
            self.b -= self.lr * (db / n)

            if epoch % max(1, self.epochs // 5) == 0 or epoch == self.epochs - 1:
                cost = sum((y_pred[i] - y[i]) ** 2 for i in range(n)) / (2 * n)
                if cost < 1e-6:
                    break

    def _predict(self, x):
        if not self._is_1d:
            return sum(w * xi for w, xi in zip(self.w, x)) + self.b
        return self.w[0] * x + self.b

    def predict(self, X):
        return [self._predict(x) for x in X]

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y))
        ss_tot = sum((yt - y_mean) ** 2 for yt in y)
        return 1 - ss_res / ss_tot


# ═══════════════════════════════════════════════
# LOGISTIC REGRESSION
# ═══════════════════════════════════════════════

def sigmoid(z):
    if z > 20:
        return 1.0
    if z < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression:
    """Binary logistic regression using gradient descent."""

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n = len(X)
        d = len(X[0])
        self.w = [0.0] * d
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = [self._predict_proba(x) for x in X]
            dw = [0.0] * d
            db = 0.0
            for i in range(n):
                error = y_pred[i] - y[i]
                for j in range(d):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(d):
                self.w[j] -= self.lr * (dw[j] / n)
            self.b -= self.lr * (db / n)

            if epoch % max(1, self.epochs // 5) == 0:
                loss = -sum(y[i] * math.log(max(y_pred[i], 1e-15))
                           + (1 - y[i]) * math.log(max(1 - y_pred[i], 1e-15))
                           for i in range(n)) / n
                if loss < 0.001:
                    break

    def _predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.w, x)) + self.b
        return sigmoid(z)

    def predict_proba(self, X):
        return [self._predict_proba(x) for x in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ═══════════════════════════════════════════════
# K-NEAREST NEIGHBORS
# ═══════════════════════════════════════════════

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(y ** 2 for y in b))
    return 1 - dot / (norm_a * norm_b)


class KNN:
    """k-Nearest Neighbors classifier."""

    def __init__(self, k=3, distance_fn=euclidean_distance):
        self.k = k
        self.distance = distance_fn

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        distances = [(self.distance(x, x_train), y_train)
                     for x_train, y_train in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda d: d[0])
        neighbors = distances[:self.k]
        votes = {}
        for _, label in neighbors:
            votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ═══════════════════════════════════════════════
# DECISION TREES
# ═══════════════════════════════════════════════

def gini_impurity(labels):
    """Gini impurity measure."""
    if not labels:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    total = len(labels)
    return 1 - sum((c / total) ** 2 for c in counts.values())


def entropy(labels):
    """Entropy of a set of labels."""
    if not labels:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


class DecisionNode:
    """A node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None):
        self.feature = feature    # index of feature to split on
        self.threshold = threshold  # value to split at
        self.left = left          # left child (<= threshold)
        self.right = right        # right child (> threshold)
        self.value = value        # predicted class (leaf only)


class DecisionTreeClassifier:
    """Classification decision tree."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return [self._traverse(x, self.root) for x in X]

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(set(y))

        # Stopping conditions
        if n_classes == 1 or n_samples < self.min_samples_split:
            return DecisionNode(value=max(set(y), key=y.count))

        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionNode(value=max(set(y), key=y.count))

        # Find best split
        best_feature, best_threshold, best_gain = None, None, 0
        parent_gini = gini_impurity(y)

        n_features = len(X[0])
        for feat in range(n_features):
            values = sorted(set(X[i][feat] for i in range(n_samples)))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left_y = []
                right_y = []
                for j in range(n_samples):
                    if X[j][feat] <= threshold:
                        left_y.append(y[j])
                    else:
                        right_y.append(y[j])

                if not left_y or not right_y:
                    continue

                gain = parent_gini - (len(left_y) / n_samples * gini_impurity(left_y)
                                      + len(right_y) / n_samples * gini_impurity(right_y))
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = threshold

        if best_gain == 0:
            return DecisionNode(value=max(set(y), key=y.count))

        # Split data
        left_X, left_y = [], []
        right_X, right_y = [], []
        for i in range(n_samples):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            left=self._build_tree(left_X, left_y, depth + 1),
            right=self._build_tree(right_X, right_y, depth + 1)
        )

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# ═══════════════════════════════════════════════
# NAIVE BAYES
# ═══════════════════════════════════════════════

class NaiveBayes:
    """Gaussian Naive Bayes classifier."""

    def fit(self, X, y):
        n = len(X)
        d = len(X[0])
        self.classes = sorted(set(y))

        # Compute priors, means, and variances
        self.priors = {}
        self.means = {c: [0.0] * d for c in self.classes}
        self.vars = {c: [0.0] * d for c in self.classes}

        for c in self.classes:
            members = [X[i] for i in range(n) if y[i] == c]
            self.priors[c] = len(members) / n
            for j in range(d):
                values = [m[j] for m in members]
                self.means[c][j] = sum(values) / len(values)
                self.vars[c][j] = (sum((v - self.means[c][j]) ** 2 for v in values)
                                   / len(values)) + 1e-9  # smoothing

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)

    def _predict_one(self, x):
        best_class = None
        best_score = -float('inf')
        for c in self.classes:
            score = math.log(self.priors[c])
            for j in range(len(x)):
                var = self.vars[c][j]
                mean = self.means[c][j]
                score += -0.5 * math.log(2 * math.pi * var) - (x[j] - mean) ** 2 / (2 * var)
            if score > best_score:
                best_score = score
                best_class = c
        return best_class


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo():
    print("🤖 Anggira — ML Fundamentals")
    print("=" * 60)

    # Generate synthetic data
    random.seed(42)
    n = 100

    # Regression data: y = 3x + 7 + noise
    X_reg = [random.gauss(0, 1) for _ in range(n)]
    y_reg = [3 * x + 7 + random.gauss(0, 1) for x in X_reg]

    # Classification data
    X_cls = []
    y_cls = []
    for i in range(n):
        label = 0 if i < n // 2 else 1
        x = [random.gauss(2, 0.5) if label == 0 else random.gauss(5, 0.5),
             random.gauss(2, 0.5) if label == 0 else random.gauss(5, 0.5)]
        X_cls.append(x)
        y_cls.append(label)

    print("\n--- Linear Regression ---")
    lr = LinearRegression(learning_rate=0.01, epochs=1000)
    lr.fit(X_reg, y_reg)
    print(f"  y = {lr.w[0]:.3f}x + {lr.b:.3f}")
    print(f"  R² = {lr.r2_score(X_reg, y_reg):.4f}")

    print("\n--- Logistic Regression ---")
    logreg = LogisticRegression()
    logreg.fit(X_cls, y_cls)
    acc = logreg.accuracy(X_cls, y_cls)
    print(f"  Accuracy: {acc:.4f}")

    print("\n--- KNN (k=3) ---")
    knn = KNN(k=3)
    knn.fit(X_cls, y_cls)
    acc_knn = knn.accuracy(X_cls, y_cls)
    print(f"  Accuracy: {acc_knn:.4f}")

    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_cls, y_cls)
    acc_dt = dt.accuracy(X_cls, y_cls)
    print(f"  Accuracy: {acc_dt:.4f}")

    print("\n--- Naive Bayes ---")
    nb = NaiveBayes()
    nb.fit(X_cls, y_cls)
    acc_nb = nb.accuracy(X_cls, y_cls)
    print(f"  Accuracy: {acc_nb:.4f}")


if __name__ == "__main__":
    demo()
