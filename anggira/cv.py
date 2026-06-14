"""
Anggira Computer Vision — Convolutions, CNNs, Augmentation, Detection

Phase: Computer Vision from Scratch
- Image representation, convolution, pooling
- CNN building blocks and a minimal CNN classifier
- Image augmentation (flips, rotations, crop, brightness/contrast)
- Run Length Encoding (RLE) for binary masks
- Naive sliding-window object detection via template matching
"""

import math
import random


# ═══════════════════════════════════════════════
# SYNTHETIC IMAGE GENERATORS (helpers)
# ═══════════════════════════════════════════════

def _normalize_2d(grid):
    """Normalize a 2D list to [0, 1]."""
    flat = [v for row in grid for v in row]
    mn, mx = min(flat), max(flat)
    if mx == mn:
        return grid
    return [[(v - mn) / (mx - mn) for v in row] for row in grid]


def _make_square(size, side=10, value=1.0):
    """Create a grayscale image with a filled square."""
    grid = [[0.0] * size for _ in range(size)]
    off = (size - side) // 2
    for y in range(off, off + side):
        for x in range(off, off + side):
            grid[y][x] = value
    return grid


def _make_circle(size, radius=None, value=1.0):
    """Create a grayscale image with a filled circle."""
    if radius is None:
        radius = size // 4
    cx = cy = size // 2
    grid = [[0.0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[y][x] = value
    return grid


def _make_line(size, thickness=2, value=1.0):
    """Create a grayscale image with a diagonal line."""
    grid = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for t in range(thickness):
            y = i
            x = i + t
            if x < size and y < size:
                grid[y][x] = value
    return grid


def demo_synthetic():
    """Return a dict of synthetic images for demonstrations."""
    return {
        'square_32': _make_square(32, 12),
        'circle_32': _make_circle(32, 10),
        'line_32': _make_line(32, 2),
    }


# ═══════════════════════════════════════════════
# IMAGE CLASS
# ═══════════════════════════════════════════════

class Image:
    """Grayscale image represented as a 2D list of floats in [0, 1]."""

    def __init__(self, grid, normalize=False):
        """
        Args:
            grid: 2D list of floats (row-major).
            normalize: if True, scale values to [0, 1].
        """
        if not grid or not grid[0]:
            raise ValueError("Image grid must be non-empty")
        self._grid = [row[:] for row in grid]
        self._h = len(grid)
        self._w = len(grid[0])
        if normalize:
            self._grid = _normalize_2d(self._grid)

    @classmethod
    def from_array(cls, array, normalize=False):
        """Create Image from a 2D list (or list of lists)."""
        return cls(array, normalize=normalize)

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    @property
    def shape(self):
        return (self._h, self._w)

    def pixel(self, x, y):
        """Get pixel value at (x, y)."""
        return self._grid[y][x]

    def set_pixel(self, x, y, value):
        """Set pixel value at (x, y)."""
        self._grid[y][x] = value

    def to_grid(self):
        """Return a copy of the internal 2D grid."""
        return [row[:] for row in self._grid]

    def __repr__(self):
        return f"Image({self._w}x{self._h}, range [{min(min(r) for r in self._grid):.3f}, {max(max(r) for r in self._grid):.3f}])"


# ═══════════════════════════════════════════════
# CONVOLUTION (2D)
# ═══════════════════════════════════════════════

def _apply_convolution(grid, kernel, stride=1, padding=0):
    """Core 2D convolution on a grid (list of lists).

    Args:
        grid: Input 2D list (H x W).
        kernel: 2D kernel (KH x KW).
        stride: Step size.
        padding: Number of zero-padding layers.

    Returns:
        2D list of convolution output.
    """
    H = len(grid)
    W = len(grid[0])
    KH = len(kernel)
    KW = len(kernel[0])

    # Pad
    if padding > 0:
        padded = [[0.0] * (W + 2 * padding) for _ in range(H + 2 * padding)]
        for y in range(H):
            for x in range(W):
                padded[y + padding][x + padding] = grid[y][x]
    else:
        padded = grid

    pH = len(padded)
    pW = len(padded[0])

    out_h = (pH - KH) // stride + 1
    out_w = (pW - KW) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Input ({pH}x{pW}) too small for kernel ({KH}x{KW}) "
            f"with stride {stride}"
        )

    output = [[0.0] * out_w for _ in range(out_h)]
    for y in range(out_h):
        for x in range(out_w):
            acc = 0.0
            for ky in range(KH):
                for kx in range(KW):
                    acc += padded[y * stride + ky][x * stride + kx] * kernel[ky][kx]
            output[y][x] = acc
    return output


class Convolution2D:
    """2D convolution with configurable kernel, stride, and padding."""

    def __init__(self, kernel, stride=1, padding=0):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def __call__(self, image):
        """Apply convolution to an Image or 2D grid."""
        if isinstance(image, Image):
            grid = image.to_grid()
        else:
            grid = image
        result = _apply_convolution(grid, self.kernel, self.stride, self.padding)
        return Image(result)

    @staticmethod
    def sobel_x():
        """Sobel X (vertical edge detection)."""
        return [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    @staticmethod
    def sobel_y():
        """Sobel Y (horizontal edge detection)."""
        return [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    @staticmethod
    def prewitt_x():
        """Prewitt X (vertical edge detection)."""
        return [[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]]

    @staticmethod
    def prewitt_y():
        """Prewitt Y (horizontal edge detection)."""
        return [[-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]]

    @staticmethod
    def gaussian(size=3, sigma=1.0):
        """Gaussian blur kernel of given size and sigma."""
        k = [[0.0] * size for _ in range(size)]
        center = size // 2
        total = 0.0
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                val = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                k[y][x] = val
                total += val
        # Normalize so kernel sums to 1
        for y in range(size):
            for x in range(size):
                k[y][x] /= total
        return k

    @staticmethod
    def identity():
        """Identity kernel (no-op)."""
        return [[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]

    @staticmethod
    def sharpen():
        """Sharpen kernel."""
        return [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]]

    @staticmethod
    def edge_detection():
        """Laplacian-like edge detection."""
        return [[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]]


def convolve(image, kernel, stride=1, padding=0):
    """Convenience function: apply convolution to an Image or grid."""
    if isinstance(image, Image):
        grid = image.to_grid()
    else:
        grid = image
    return Image(_apply_convolution(grid, kernel, stride, padding))


def sobel_edge_magnitude(image):
    """Compute Sobel edge magnitude from an Image or grid.

    Returns an Image where each pixel is sqrt(Gx^2 + Gy^2).
    """
    gx = convolve(image, Convolution2D.sobel_x()).to_grid()
    gy = convolve(image, Convolution2D.sobel_y()).to_grid()
    H, W = len(gx), len(gx[0])
    mag = [[0.0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            mag[y][x] = math.sqrt(gx[y][x] ** 2 + gy[y][x] ** 2)
    return Image(mag, normalize=True)


# ═══════════════════════════════════════════════
# POOLING
# ═══════════════════════════════════════════════

def _pool(grid, kernel_size, stride, mode='max'):
    """Generic pooling operation."""
    H = len(grid)
    W = len(grid[0])
    K = kernel_size
    if stride is None:
        stride = K

    out_h = (H - K) // stride + 1
    out_w = (W - K) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Pooling: input ({H}x{W}) too small for kernel {K}, stride {stride}")

    out = [[0.0] * out_w for _ in range(out_h)]
    for y in range(out_h):
        for x in range(out_w):
            window = [
                grid[y * stride + ky][x * stride + kx]
                for ky in range(K) for kx in range(K)
            ]
            if mode == 'max':
                out[y][x] = max(window)
            else:  # avg
                out[y][x] = sum(window) / len(window)
    return out


class MaxPool2D:
    """Max pooling over a 2D grid or Image."""

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, image):
        if isinstance(image, Image):
            grid = image.to_grid()
        else:
            grid = image
        return Image(_pool(grid, self.kernel_size, self.stride, 'max'))


class AvgPool2D:
    """Average pooling over a 2D grid or Image."""

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, image):
        if isinstance(image, Image):
            grid = image.to_grid()
        else:
            grid = image
        return Image(_pool(grid, self.kernel_size, self.stride, 'avg'))


# ═══════════════════════════════════════════════
# ACTIVATION (ReLU)
# ═══════════════════════════════════════════════

def relu(x):
    """ReLU activation for a scalar."""
    return max(0.0, x)


def relu_2d(grid):
    """Apply ReLU element-wise to a 2D grid."""
    return [[max(0.0, v) for v in row] for row in grid]


# ═══════════════════════════════════════════════
# CNN BLOCK
# ═══════════════════════════════════════════════

class CNNBlock:
    """Conv2D → ReLU → Pool as a building block."""

    def __init__(self, kernel, pool_mode='max', pool_size=2, pool_stride=None,
                 conv_stride=1, conv_padding=0):
        """
        Args:
            kernel: Convolution kernel (2D list).
            pool_mode: 'max' or 'avg'.
            pool_size: Pooling kernel size.
            pool_stride: Pooling stride (default = pool_size).
            conv_stride: Convolution stride.
            conv_padding: Convolution padding.
        """
        self.conv = Convolution2D(kernel, stride=conv_stride, padding=conv_padding)
        if pool_mode == 'max':
            self.pool = MaxPool2D(pool_size, pool_stride)
        else:
            self.pool = AvgPool2D(pool_size, pool_stride)

    def forward(self, image):
        """Run Conv2D → ReLU → Pool."""
        out = self.conv(image)
        grid = relu_2d(out.to_grid())
        out = self.pool(Image(grid))
        return out

    def __call__(self, image):
        return self.forward(image)


# ═══════════════════════════════════════════════
# SIMPLE CNN CLASSIFIER
# ═══════════════════════════════════════════════

def _flatten(grid):
    """Flatten a 2D grid into a 1D list."""
    return [v for row in grid for v in row]


def _softmax(logits):
    """Numerically stable softmax."""
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _cross_entropy_loss(probs, target):
    """Cross-entropy loss (target is integer class index)."""
    eps = 1e-15
    return -math.log(max(probs[target], eps))


class SimpleCNN:
    """A small CNN (2 conv blocks + 1 fully-connected layer) for classification.

    Designed for synthetic 2D shapes on small images (e.g., 32x32).
    Architecture:
      Block 1: conv (3x3, pad=1) → ReLU → maxpool 2x2
      Block 2: conv (3x3, pad=1) → ReLU → maxpool 2x2
      FC: flatten → linear → num_classes
    """

    def __init__(self, num_classes=3, hidden_dim=64):
        """
        Args:
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension for the FC layer.
        """
        # Block 1 kernel: simple edge filter
        self.block1 = CNNBlock(
            kernel=Convolution2D.sobel_x(),
            pool_mode='max', pool_size=2,
            conv_padding=1
        )
        # Block 2 kernel: another edge orientation
        self.block2 = CNNBlock(
            kernel=Convolution2D.sobel_y(),
            pool_mode='max', pool_size=2,
            conv_padding=1
        )
        # FC weights and bias (lazy init)
        self.fc_w = None   # set by _init_fc or during forward
        self.fc_b = None
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self._fc_in_features = None

    def _init_fc(self, in_features):
        """Initialize FC weights with Xavier-like init."""
        self._fc_in_features = in_features
        # Xavier/Glorot init
        limit = math.sqrt(6.0 / (in_features + self.hidden_dim))
        self.fc_w = [[random.uniform(-limit, limit) for _ in range(in_features)]
                     for _ in range(self.hidden_dim)]
        self.fc_b = [0.0] * self.hidden_dim

        limit2 = math.sqrt(6.0 / (self.hidden_dim + self.num_classes))
        self.fc_out_w = [[random.uniform(-limit2, limit2) for _ in range(self.hidden_dim)]
                         for _ in range(self.num_classes)]
        self.fc_out_b = [0.0] * self.num_classes

    def forward(self, image):
        """Forward pass.

        Args:
            image: Image instance or 2D grid.

        Returns:
            logits (list of floats) for each class.
        """
        if isinstance(image, Image):
            grid = image.to_grid()
        else:
            grid = image

        # Block 1
        x = Image(grid)
        x = self.block1(x)  # Conv → ReLU → Pool
        # Block 2
        x = self.block2(x)

        flat = _flatten(x.to_grid())
        in_features = len(flat)

        if self.fc_w is None:
            self._init_fc(in_features)
        elif self._fc_in_features != in_features:
            self._init_fc(in_features)

        # Hidden layer
        hidden = [sum(w[j] * flat[j] for j in range(in_features)) + self.fc_b[i]
                  for i, w in enumerate(self.fc_w)]
        hidden = [max(0.0, h) for h in hidden]  # ReLU

        # Output layer
        logits = [sum(self.fc_out_w[i][j] * hidden[j] for j in range(self.hidden_dim))
                  + self.fc_out_b[i]
                  for i in range(self.num_classes)]
        return logits

    def predict(self, image):
        """Return predicted class index and confidence."""
        logits = self.forward(image)
        probs = _softmax(logits)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        return pred, probs[pred]

    def train_step(self, image, target, lr=0.01):
        """One training step via numerical gradients.

        Args:
            image: Image or 2D grid input.
            target: Integer class index.
            lr: Learning rate.

        Returns:
            loss (float) before update.
        """
        if isinstance(image, Image):
            grid = image.to_grid()
        else:
            grid = image

        x = Image(grid)
        # Get intermediate values for gradient computation
        x1 = self.block1(x)
        x2 = self.block2(x1)
        flat = _flatten(x2.to_grid())
        in_features = len(flat)

        if self.fc_w is None:
            self._init_fc(in_features)

        # Forward to logits
        hidden = [sum(self.fc_w[i][j] * flat[j] for j in range(in_features))
                  + self.fc_b[i] for i in range(self.hidden_dim)]
        hidden_relu = [max(0.0, h) for h in hidden]
        logits = [sum(self.fc_out_w[i][j] * hidden_relu[j] for j in range(self.hidden_dim))
                  + self.fc_out_b[i] for i in range(self.num_classes)]
        probs = _softmax(logits)
        loss = _cross_entropy_loss(probs, target)

        # Gradient of loss w.r.t. logits
        d_logits = [p - (1.0 if i == target else 0.0) for i, p in enumerate(probs)]

        # Gradients for fc_out_w and fc_out_b
        grad_fc_out_w = [[0.0] * self.hidden_dim for _ in range(self.num_classes)]
        grad_fc_out_b = [0.0] * self.num_classes
        for i in range(self.num_classes):
            for j in range(self.hidden_dim):
                grad_fc_out_w[i][j] = d_logits[i] * hidden_relu[j]
            grad_fc_out_b[i] = d_logits[i]

        # Gradient through output layer to hidden
        d_hidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            for i in range(self.num_classes):
                d_hidden[j] += d_logits[i] * self.fc_out_w[i][j]
        # ReLU derivative
        d_hidden_relu = [d_hidden[j] * (1.0 if hidden[j] > 0 else 0.0)
                         for j in range(self.hidden_dim)]

        # Gradients for fc_w and fc_b
        grad_fc_w = [[0.0] * in_features for _ in range(self.hidden_dim)]
        grad_fc_b = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            for j in range(in_features):
                grad_fc_w[i][j] = d_hidden_relu[i] * flat[j]
            grad_fc_b[i] = d_hidden_relu[i]

        # SGD update
        for i in range(self.num_classes):
            for j in range(self.hidden_dim):
                self.fc_out_w[i][j] -= lr * grad_fc_out_w[i][j]
            self.fc_out_b[i] -= lr * grad_fc_out_b[i]
        for i in range(self.hidden_dim):
            for j in range(in_features):
                self.fc_w[i][j] -= lr * grad_fc_w[i][j]
            self.fc_b[i] -= lr * grad_fc_b[i]

        return loss


# ═══════════════════════════════════════════════
# IMAGE AUGMENTATION
# ═══════════════════════════════════════════════

class ImageAugmentation:
    """Random augmentations for 2D list images (no PIL)."""

    @staticmethod
    def flip_horizontal(grid):
        """Horizontal flip (mirror left-right)."""
        return [row[::-1] for row in grid]

    @staticmethod
    def flip_vertical(grid):
        """Vertical flip (mirror top-bottom)."""
        return grid[::-1]

    @staticmethod
    def rotate_90(grid):
        """Rotate 90 degrees clockwise."""
        H = len(grid)
        W = len(grid[0])
        return [[grid[H - 1 - y][x] for y in range(H)] for x in range(W)]

    @staticmethod
    def rotate_180(grid):
        """Rotate 180 degrees."""
        return [row[::-1] for row in grid[::-1]]

    @staticmethod
    def rotate_270(grid):
        """Rotate 270 degrees clockwise (or 90 counter-clockwise)."""
        H = len(grid)
        W = len(grid[0])
        return [[grid[y][W - 1 - x] for y in range(H)] for x in range(W)]

    @staticmethod
    def crop(grid, x, y, w, h):
        """Crop a region from the grid.

        Args:
            x, y: Top-left corner.
            w, h: Width and height of crop region.
        """
        return [row[x:x + w] for row in grid[y:y + h]]

    @staticmethod
    def adjust_brightness(grid, factor):
        """Adjust brightness: multiply every pixel by factor."""
        return [[min(1.0, max(0.0, v * factor)) for v in row] for row in grid]

    @staticmethod
    def adjust_contrast(grid, factor):
        """Adjust contrast: (v - 0.5) * factor + 0.5, clipped to [0, 1]."""
        return [[min(1.0, max(0.0, (v - 0.5) * factor + 0.5)) for v in row]
                for row in grid]

    def random_transform(self, grid, seed=None):
        """Apply a random sequence of augmentations.

        Args:
            grid: Input 2D grid.
            seed: Optional random seed for reproducibility.

        Returns:
            Augmented 2D grid.
        """
        if seed is not None:
            random.seed(seed)
        aug = grid

        # Random horizontal flip (50%)
        if random.random() < 0.5:
            aug = self.flip_horizontal(aug)

        # Random vertical flip (50%)
        if random.random() < 0.5:
            aug = self.flip_vertical(aug)

        # Random rotation (25% each: 0, 90, 180, 270)
        r = random.choice([0, 90, 180, 270])
        if r == 90:
            aug = self.rotate_90(aug)
        elif r == 180:
            aug = self.rotate_180(aug)
        elif r == 270:
            aug = self.rotate_270(aug)

        # Random brightness jitter (80% chance)
        if random.random() < 0.8:
            factor = random.uniform(0.5, 1.5)
            aug = self.adjust_brightness(aug, factor)

        # Random contrast jitter (80% chance)
        if random.random() < 0.8:
            factor = random.uniform(0.5, 1.5)
            aug = self.adjust_contrast(aug, factor)

        return aug


# ═══════════════════════════════════════════════
# RUN LENGTH ENCODING (RLE)
# ═══════════════════════════════════════════════

def rle_encode(mask):
    """Encode a binary mask (2D list of 0/1 values) into RLE.

    Returns a list of (value, count) pairs, scanning row-major.
    """
    if not mask or not mask[0]:
        return []
    encoded = []
    current_val = mask[0][0]
    count = 0
    for row in mask:
        for v in row:
            if v == current_val:
                count += 1
            else:
                encoded.append((current_val, count))
                current_val = v
                count = 1
    encoded.append((current_val, count))
    return encoded


def rle_decode(encoded, height, width):
    """Decode RLE (list of (value, count) pairs) back into a 2D binary mask.

    Args:
        encoded: RLE encoding from rle_encode().
        height, width: Dimensions of the output mask.

    Returns:
        2D list (height x width) of floats (0.0 or 1.0).
    """
    mask = [[0.0] * width for _ in range(height)]
    idx = 0
    for val, count in encoded:
        for _ in range(count):
            y = idx // width
            x = idx % width
            if y >= height:
                break
            mask[y][x] = float(val)
            idx += 1
    return mask


def rle_area(encoded):
    """Compute the area (number of '1' pixels) from RLE encoding."""
    return sum(count for val, count in encoded if val == 1)


def rle_compress(encoded):
    """Merge adjacent runs of the same value in RLE (already guaranteed by encode)."""
    return encoded  # rle_encode already produces merged runs


# ═══════════════════════════════════════════════
# NAIVE OBJECT DETECTION (sliding window + NCC)
# ═══════════════════════════════════════════════

def _normalized_cross_correlation(patch, template):
    """Compute normalized cross-correlation between two same-size 2D grids."""
    H = len(patch)
    W = len(patch[0])
    n = H * W

    # Flatten
    p_flat = [patch[y][x] for y in range(H) for x in range(W)]
    t_flat = [template[y][x] for y in range(H) for x in range(W)]

    mean_p = sum(p_flat) / n
    mean_t = sum(t_flat) / n

    p_dev = [v - mean_p for v in p_flat]
    t_dev = [v - mean_t for v in t_flat]

    num = sum(p_dev[i] * t_dev[i] for i in range(n))
    denom_p = math.sqrt(sum(v * v for v in p_dev))
    denom_t = math.sqrt(sum(v * v for v in t_dev))

    # Edge case: both are constant (zero variance) -> perfect match if same constant
    if denom_p == 0 and denom_t == 0:
        return 1.0 if mean_p == mean_t else 0.0
    if denom_p == 0 or denom_t == 0:
        return 0.0

    return num / (denom_p * denom_t)


def sliding_window_detect(image, template, stride=1, threshold=0.6):
    """Naive sliding-window object detection using normalized cross-correlation.

    Args:
        image: Image or 2D grid (must be larger than template).
        template: Image or 2D grid (the pattern to find).
        stride: Step size for the sliding window.
        threshold: NCC threshold [0, 1]; detections above this are reported.

    Returns:
        List of (x, y, score) tuples, where (x, y) is the top-left corner
        of each detection window.
    """
    if isinstance(image, Image):
        img_grid = image.to_grid()
    else:
        img_grid = image

    if isinstance(template, Image):
        tpl_grid = template.to_grid()
    else:
        tpl_grid = template

    img_h = len(img_grid)
    img_w = len(img_grid[0])
    tpl_h = len(tpl_grid)
    tpl_w = len(tpl_grid[0])

    if img_h < tpl_h or img_w < tpl_w:
        raise ValueError("Image must be larger than template in both dimensions")

    detections = []
    for y in range(0, img_h - tpl_h + 1, stride):
        for x in range(0, img_w - tpl_w + 1, stride):
            patch = [row[x:x + tpl_w] for row in img_grid[y:y + tpl_h]]
            score = _normalized_cross_correlation(patch, tpl_grid)
            if score >= threshold:
                detections.append((x, y, score))

    # Sort by score descending
    detections.sort(key=lambda d: d[2], reverse=True)

    # Non-maximum suppression (simple: skip overlapping windows)
    kept = []
    for x, y, s in detections:
        overlap = False
        for kx, ky, _ in kept:
            ix = max(x, kx)
            iy = max(y, ky)
            iw = min(x + tpl_w, kx + tpl_w) - ix
            ih = min(y + tpl_h, ky + tpl_h) - iy
            if iw > 0 and ih > 0:
                inter = iw * ih
                union = tpl_w * tpl_h * 2 - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.3:
                    overlap = True
                    break
        if not overlap:
            kept.append((x, y, s))

    return kept


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo():
    """Demonstrate all components of the CV module."""
    print("=" * 60)
    print("ANGGIRA COMPUTER VISION DEMO")
    print("=" * 60)

    # ── 1. Image class ────────────────────────────
    print("\n1. IMAGE CLASS")
    raw = _make_square(8, 4, 1.0)
    img = Image(raw)
    print(f"   Created Image: {img}")
    print(f"   Width: {img.width}, Height: {img.height}, Shape: {img.shape}")
    print(f"   Pixel at (0,0): {img.pixel(0,0):.1f}")
    print(f"   Pixel at (2,2): {img.pixel(2,2):.1f}")
    img2 = Image.from_array(raw, normalize=True)
    print(f"   From array (normalized): {img2}")

    # ── 2. Convolution ────────────────────────────
    print("\n2. CONVOLUTION")
    square_16 = _make_square(16, 8, 1.0)
    img_sq = Image(square_16)

    # Gaussian blur
    gauss_k = Convolution2D.gaussian(3, 0.8)
    blur = Convolution2D(gauss_k)
    blurred = blur(img_sq)
    print(f"   Gaussian blur: {blurred}")

    # Sobel edge detection
    sobel_x = Convolution2D(Convolution2D.sobel_x())
    edges_x = sobel_x(img_sq)
    print(f"   Sobel X edges: {edges_x}")

    # Sobel magnitude
    mag = sobel_edge_magnitude(img_sq)
    print(f"   Sobel magnitude: {mag}")

    # Prewitt
    prew = Convolution2D(Convolution2D.prewitt_x())
    prew_out = prew(img_sq)
    print(f"   Prewitt X: {prew_out}")

    # Sharpen
    sharp = Convolution2D(Convolution2D.sharpen())
    sharp_out = sharp(img_sq)
    print(f"   Sharpen: {sharp_out}")

    # Identity (with padding=1 to preserve size)
    ident = Convolution2D(Convolution2D.identity(), padding=1)
    ident_out = ident(img_sq)
    diff = sum(abs(ident_out.to_grid()[y][x] - square_16[y][x])
               for y in range(16) for x in range(16))
    print(f"   Identity (pad=1): total pixel diff = {diff:.6f} (0 = perfect)")

    # ── 3. Pooling ────────────────────────────────
    print("\n3. POOLING")
    pool_max = MaxPool2D(2)
    pooled_max = pool_max(img_sq)
    print(f"   MaxPool 2x2: {pooled_max} (size {pooled_max.width}x{pooled_max.height})")

    pool_avg = AvgPool2D(2)
    pooled_avg = pool_avg(img_sq)
    print(f"   AvgPool 2x2: {pooled_avg} (size {pooled_avg.width}x{pooled_avg.height})")

    # ── 4. CNN Block ──────────────────────────────
    print("\n4. CNN BLOCK")
    block = CNNBlock(
        kernel=Convolution2D.sobel_x(),
        pool_mode='max', pool_size=2
    )
    block_out = block(img_sq)
    print(f"   CNNBlock output: {block_out} (size {block_out.width}x{block_out.height})")

    # ── 5. SimpleCNN ──────────────────────────────
    print("\n5. SIMPLE CNN (Training on synthetic shapes)")
    cnn = SimpleCNN(num_classes=3)

    # Create 3 synthetic classes: square, circle, line
    train_data = []
    for _ in range(50):
        train_data.append((_make_square(32, random.randint(6, 14), 1.0), 0))
        train_data.append((_make_circle(32, random.randint(5, 12), 1.0), 1))
        train_data.append((_make_line(32, random.randint(1, 3), 1.0), 2))

    random.shuffle(train_data)
    print(f"   Training samples: {len(train_data)}")

    for epoch in range(5):
        total_loss = 0.0
        correct = 0
        for grid, label in train_data:
            loss = cnn.train_step(Image(grid), label, lr=0.05)
            total_loss += loss
            pred, conf = cnn.predict(Image(grid))
            if pred == label:
                correct += 1
        acc = correct / len(train_data) * 100
        print(f"   Epoch {epoch + 1}: loss={total_loss / len(train_data):.4f}, "
              f"acc={acc:.1f}%")

    # Test predictions
    test_sq = _make_square(32, 10, 1.0)
    pred, conf = cnn.predict(Image(test_sq))
    print(f"   Square -> class {pred} (confidence {conf:.3f})")
    test_circ = _make_circle(32, 8, 1.0)
    pred, conf = cnn.predict(Image(test_circ))
    print(f"   Circle -> class {pred} (confidence {conf:.3f})")
    test_ln = _make_line(32, 2, 1.0)
    pred, conf = cnn.predict(Image(test_ln))
    print(f"   Line   -> class {pred} (confidence {conf:.3f})")

    # ── 6. Image Augmentation ─────────────────────
    print("\n6. IMAGE AUGMENTATION")
    aug = ImageAugmentation()
    grid_in = _make_square(16, 6, 1.0)

    hf = aug.flip_horizontal(grid_in)
    print(f"   Horizontal flip: {Image(hf)}")

    vf = aug.flip_vertical(grid_in)
    print(f"   Vertical flip: {Image(vf)}")

    r90 = aug.rotate_90(grid_in)
    print(f"   Rotate 90: shape {len(r90)}x{len(r90[0])}")

    r180 = aug.rotate_180(grid_in)
    print(f"   Rotate 180: shape {len(r180)}x{len(r180[0])}")

    r270 = aug.rotate_270(grid_in)
    print(f"   Rotate 270: shape {len(r270)}x{len(r270[0])}")

    crop = aug.crop(grid_in, 2, 2, 8, 8)
    print(f"   Crop (2,2,8,8): {Image(crop)}")

    bright = aug.adjust_brightness(grid_in, 0.5)
    print(f"   Brightness x0.5: range [{min(min(r) for r in bright):.2f}, "
          f"{max(max(r) for r in bright):.2f}]")

    contrast = aug.adjust_contrast(grid_in, 2.0)
    print(f"   Contrast x2.0: range [{min(min(r) for r in contrast):.2f}, "
          f"{max(max(r) for r in contrast):.2f}]")

    rand_aug = aug.random_transform(grid_in, seed=42)
    print(f"   Random transform: {Image(rand_aug)}")

    # ── 7. RLE ────────────────────────────────────
    print("\n7. RUN LENGTH ENCODING (RLE)")
    mask = [
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ]
    enc = rle_encode(mask)
    print(f"   Original mask size: {len(mask)}x{len(mask[0])}")
    print(f"   Encoded RLE: {enc}")
    print(f"   Foreground area: {rle_area(enc)} pixels")
    decoded = rle_decode(enc, 4, 5)
    print(f"   Decoded matches original: {decoded == mask}")

    # Larger mask test
    big_mask = [[1 if (x - 8) ** 2 + (y - 8) ** 2 < 25 else 0
                 for x in range(16)] for y in range(16)]
    big_enc = rle_encode(big_mask)
    big_dec = rle_decode(big_enc, 16, 16)
    print(f"   16x16 circle mask RLE: {len(big_enc)} runs (vs 256 pixels), "
          f"area={rle_area(big_enc)}, "
          f"reversible={big_dec == big_mask}")

    # ── 8. Object Detection ───────────────────────
    print("\n8. OBJECT DETECTION (Sliding Window + NCC)")
    # Create a 32x32 scene with a square somewhere
    scene = [[0.0] * 32 for _ in range(32)]
    for y in range(8, 18):
        for x in range(12, 22):
            scene[y][x] = 1.0
    # Add a noisy background pixel
    scene[3][28] = 0.5

    # Template: a 10x10 solid square (matches the one placed in the scene)
    template = [[1.0 for _ in range(10)] for _ in range(10)]

    detections = sliding_window_detect(scene, template, stride=1, threshold=0.5)
    print(f"   Found {len(detections)} detection(s):")
    for x, y, s in detections[:5]:
        print(f"     Top-left=({x},{y}), score={s:.3f}")
    if detections:
        print(f"   Best detection centered at ({detections[0][0] + 5}, {detections[0][1] + 5}) "
              f"— near expected center (17, 13)")

    # ── 9. Synthetic data ─────────────────────────
    print("\n9. SYNTHETIC IMAGES")
    syn = demo_synthetic()
    for name, data in syn.items():
        img_syn = Image(data)
        print(f"   {name}: {img_syn}")

    print("\n" + "=" * 60)
    print("CV MODULE DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    demo()
