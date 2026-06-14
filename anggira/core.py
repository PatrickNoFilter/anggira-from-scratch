"""
Anggira Core — Vector, Matrix, and Linear Algebra Foundation

Phase 1: Math Foundations (Lessons 1-3)
- Vectors, Matrices, Transformations, Eigenstuff
"""

import math
import random


# ═══════════════════════════════════════════════
# VECTORS
# ═══════════════════════════════════════════════

class Vector:
    """A vector in n-dimensional space — the atom of AI."""

    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.components])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot(self, other):
        """Dot product — measures similarity between vectors."""
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        """Euclidean norm (L2)."""
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        """Unit vector in the same direction."""
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        """Cosine similarity between [-1, 1]. Used everywhere in AI."""
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        """Angle in degrees between two vectors."""
        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp numerical errors
        return math.degrees(math.acos(cos_theta))

    def project_onto(self, other):
        """Project this vector onto another."""
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        if self.dim <= 6:
            return f"Vector({self.components})"
        return f"Vector(dim={self.dim}, first={self.components[:3]}...)"


# ═══════════════════════════════════════════════
# MATRICES
# ═══════════════════════════════════════════════

class Matrix:
    """A matrix — transformation, weights, or data."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __add__(self, other):
        return Matrix([
            [self.rows[i][j] + other.rows[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ])

    def __sub__(self, other):
        return Matrix([
            [self.rows[i][j] - other.rows[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ])

    def __mul__(self, scalar):
        return Matrix([
            [x * scalar for x in row] for row in self.rows
        ])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def element_mul(self, other):
        """Element-wise (Hadamard) product."""
        return Matrix([
            [self.rows[i][j] * other.rows[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ])

    def __matmul__(self, other):
        """Matrix multiplication — the core operation of neural networks."""
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def det_2x2(self):
        assert self.shape == (2, 2), "Only 2x2"
        return self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]

    def det_3x3(self):
        assert self.shape == (3, 3), "Only 3x3"
        m = self.rows
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    def inverse_2x2(self):
        assert self.shape == (2, 2), "Only 2x2"
        det = self.det_2x2()
        assert abs(det) > 1e-10, "Matrix is singular"
        a, b = self.rows[0]
        c, d = self.rows[1]
        return Matrix([[d / det, -b / det], [-c / det, a / det]])

    def rank(self):
        """Row rank via Gaussian elimination."""
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def eigenvalues_2x2(self):
        """Characteristic polynomial: λ² - tr(A)λ + det(A) = 0"""
        a, b = self.rows[0]
        c, d = self.rows[1]
        trace = a + d
        det = a * d - b * c
        disc = trace ** 2 - 4 * det
        if disc < 0:
            real = trace / 2
            imag = (-disc) ** 0.5 / 2
            return (complex(real, imag), complex(real, -imag))
        sqrt_disc = disc ** 0.5
        return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

    def eigenvector_2x2(self, eigenvalue):
        """Compute eigenvector for a given eigenvalue (2x2)."""
        a, b = self.rows[0]
        c, d = self.rows[1]
        if abs(b) > 1e-10:
            v = [b, eigenvalue - a]
        elif abs(c) > 1e-10:
            v = [eigenvalue - d, c]
        else:
            v = [1, 0] if abs(a - eigenvalue) < 1e-10 else [0, 1]
        mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return [v[0] / mag, v[1] / mag]

    def __repr__(self):
        return f"Matrix({self.rows})"


# ═══════════════════════════════════════════════
# TRANSFORMATIONS
# ═══════════════════════════════════════════════

def rotation_2d(theta):
    """2D rotation matrix (counter-clockwise by theta radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return Matrix([[c, -s], [s, c]])


def rotation_3d_z(theta):
    """3D rotation around z-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_3d_x(theta):
    """3D rotation around x-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_3d_y(theta):
    """3D rotation around y-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def scaling_2d(sx, sy):
    return Matrix([[sx, 0], [0, sy]])


def shearing_2d(kx, ky):
    return Matrix([[1, kx], [ky, 1]])


def reflection_x():
    return Matrix([[1, 0], [0, -1]])


def reflection_y():
    return Matrix([[-1, 0], [0, 1]])


# ═══════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════

def is_independent(vectors):
    """Check if a set of vectors is linearly independent."""
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def gram_schmidt(vectors):
    """Gram-Schmidt process: orthogonalize a set of vectors."""
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


# ═══════════════════════════════════════════════
# DEMO / VERIFICATION
# ═══════════════════════════════════════════════

def demo():
    print("🤖 Anggira — Core Math Foundations Demo")
    print("=" * 60)

    print("\n--- Vectors ---")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  a + b = {a + b}")
    print(f"  a · b (dot) = {a.dot(b)}")
    print(f"  |a| = {a.magnitude():.4f}")
    print(f"  cosine_sim(a,b) = {a.cosine_similarity(b):.4f}")
    print(f"  angle(a,b) = {a.angle_between(b):.1f}°")

    print("\n--- Matrix Transformations ---")
    R = rotation_2d(math.pi / 4)
    point = Vector([1, 0])
    rotated = R @ point
    print(f"  Rotate (1,0) by 45°: {rotated}")

    print("\n--- Eigen Decomposition ---")
    A = Matrix([[3, 1], [0, 2]])
    vals = A.eigenvalues_2x2()
    print(f"  A = {A}")
    print(f"  Eigenvalues: {vals}")
    for v in vals:
        vec = A.eigenvector_2x2(v)
        print(f"  Eigenvector for λ={v}: {vec}")

    print("\n--- Neural Network Layer ---")
    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
    input_vec = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vec
    print(f"  Input (3D):  {input_vec}")
    print(f"  Output (2D): {output}")
    print("  ^ This is literally what a neural network layer does.")


if __name__ == "__main__":
    demo()
