"""
Anggira Audio — Speech & Audio Processing from Scratch

Phase: Speech & Audio
  - AudioGenerator: sine, square, chirp synthesis
  - FFT/DFT: Cooley-Tukey radix-2 FFT
  - STFT: framing + windowing + FFT per frame
  - MelFilterBank: triangular filters on mel scale
  - MFCC: DCT of log-mel spectrogram
  - VAD: energy-based + spectral flatness voice activity detection
  - AudioCodec: simple RVQ-like quantization with multiple codebooks

Pure Python (stdlib only — math, cmath, random).
"""

import math
import cmath
import random

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

TAU = 2.0 * math.pi


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ═══════════════════════════════════════════════════════════════
# 1 — AUDIO GENERATOR
# ═══════════════════════════════════════════════════════════════

class AudioGenerator:
    """Synthesize audio waveforms from scratch."""

    @staticmethod
    def sine(frequency: float, duration: float,
             sample_rate: int = 16000, amplitude: float = 0.5) -> list[float]:
        """Generate a sine wave.

        Args:
            frequency: Frequency in Hz.
            duration: Duration in seconds.
            sample_rate: Samples per second.
            amplitude: Peak amplitude (0.0 to 1.0).

        Returns:
            list[float]: Normalised samples in [-1, 1].
        """
        n = int(sample_rate * duration)
        return [amplitude * math.sin(TAU * frequency * t / sample_rate)
                for t in range(n)]

    @staticmethod
    def square(frequency: float, duration: float,
               sample_rate: int = 16000, amplitude: float = 0.5,
               duty: float = 0.5) -> list[float]:
        """Generate a square wave.

        Args:
            frequency: Frequency in Hz.
            duration: Duration in seconds.
            sample_rate: Samples per second.
            amplitude: Peak amplitude.
            duty: Duty cycle (0.0 to 1.0, default 0.5).

        Returns:
            list[float]: Samples in [-amplitude, amplitude].
        """
        n = int(sample_rate * duration)
        period_samples = sample_rate / frequency
        return [
            amplitude if (t % period_samples) / period_samples < duty
            else -amplitude
            for t in range(n)
        ]

    @staticmethod
    def chirp(f0: float, f1: float, duration: float,
              sample_rate: int = 16000, amplitude: float = 0.5) -> list[float]:
        """Generate a linear frequency chirp (frequency sweep).

        Instantaneous frequency goes linearly from f0 to f1.

        Args:
            f0: Start frequency in Hz.
            f1: End frequency in Hz.
            duration: Duration in seconds.
            sample_rate: Samples per second.
            amplitude: Peak amplitude.

        Returns:
            list[float]: Normalised samples.
        """
        n = int(sample_rate * duration)
        rate = (f1 - f0) / duration       # Hz per second
        samples = []
        for t in range(n):
            sec = t / sample_rate
            # phase = ∫(f0 + rate·τ) dτ = f0·t + ½·rate·t²
            phase = TAU * (f0 * sec + 0.5 * rate * sec * sec)
            samples.append(amplitude * math.sin(phase))
        return samples

    @staticmethod
    def silence(duration: float, sample_rate: int = 16000) -> list[float]:
        """Generate silence (all zeros)."""
        return [0.0] * int(sample_rate * duration)

    @staticmethod
    def normalize(samples: list[float]) -> list[float]:
        """Normalise samples to [-1.0, 1.0] in-place copy."""
        if not samples:
            return samples
        peak = max(abs(s) for s in samples)
        return [s / peak for s in samples] if peak > 1e-12 else samples[:]


# ═══════════════════════════════════════════════════════════════
# 2 — FFT / DFT  (Cooley–Tukey radix-2)
# ═══════════════════════════════════════════════════════════════

def fft(x: list) -> list[complex]:
    """Fast Fourier Transform via Cooley–Tukey radix-2.

    If len(x) is not a power of two, the signal is zero-padded to the
    next power of two before the transform.

    Args:
        x: list of real or complex samples.

    Returns:
        list[complex]: FFT output (length = next power of two).
    """
    x = [complex(v) for v in x]
    n = len(x)
    n2 = _next_power_of_two(n)
    if n2 > n:
        x = x + [0j] * (n2 - n)
    return _fft_recursive(x)


def _fft_recursive(x: list[complex]) -> list[complex]:
    """Cooley–Tukey radix-2 FFT (recursive).  n must be a power of two."""
    n = len(x)
    if n == 1:
        return list(x)

    even = _fft_recursive(x[0::2])
    odd  = _fft_recursive(x[1::2])

    out = [0j] * n
    half = n // 2
    for k in range(half):
        twiddle = cmath.exp(complex(0, -TAU * k / n))
        t = twiddle * odd[k]
        out[k]         = even[k] + t
        out[k + half]  = even[k] - t
    return out


def ifft(X: list[complex]) -> list[complex]:
    """Inverse FFT via the standard conjugation trick.

    Args:
        X: list of complex frequency bins.

    Returns:
        list[complex]: Time-domain signal.
    """
    n = len(X)
    conj = [v.conjugate() for v in X]
    result = fft(conj)
    return [v.conjugate() / n for v in result]


# ── Reference DFT  (O(n²) — for verification) ─────────────────

def dft(x: list) -> list[complex]:
    """Direct DFT — O(n²) reference implementation.

    Works for any length (no zero-padding).
    """
    x = [complex(v) for v in x]
    n = len(x)
    return [
        sum(x[t] * cmath.exp(complex(0, -TAU * k * t / n)) for t in range(n))
        for k in range(n)
    ]


def idft(X: list[complex]) -> list[complex]:
    """Inverse DFT — O(n²) reference."""
    n = len(X)
    return [
        sum(X[k] * cmath.exp(complex(0, TAU * k * t / n)) for k in range(n))
        / n for t in range(n)
    ]


# ── Spectral helpers ──────────────────────────────────────────

def magnitude_spectrum(X: list[complex]) -> list[float]:
    """Magnitude spectrum (positive frequencies only: bins 0 .. n//2)."""
    half = len(X) // 2
    return [abs(v) for v in X[:half + 1]]


def power_spectrum(X: list[complex]) -> list[float]:
    """Power spectrum (positive frequencies only)."""
    return [m * m for m in magnitude_spectrum(X)]


# ═══════════════════════════════════════════════════════════════
# 3 — WINDOWING
# ═══════════════════════════════════════════════════════════════

def hann_window(size: int) -> list[float]:
    """Hann window.

    w[n] = 0.5 * (1 - cos(2π n / (N-1))),  n = 0 … N-1
    """
    if size <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(TAU * n / (size - 1))) for n in range(size)]


def hamming_window(size: int) -> list[float]:
    """Hamming window.

    w[n] = 0.54 - 0.46 * cos(2π n / (N-1)),  n = 0 … N-1
    """
    if size <= 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(TAU * n / (size - 1)) for n in range(size)]


# ═══════════════════════════════════════════════════════════════
# 4 — STFT
# ═══════════════════════════════════════════════════════════════

def stft(samples: list[float], frame_size: int = 1024,
         hop_size: int = 512, window_type: str = 'hann'
         ) -> tuple[list[list[complex]], list[list[float]]]:
    """Short-Time Fourier Transform.

    Steps:
      1. Frame the signal (overlapping frames).
      2. Multiply each frame by the chosen window.
      3. Apply FFT to each windowed frame.

    Args:
        samples: Input audio.
        frame_size: Samples per frame (power of two recommended).
        hop_size:  Samples between frame starts.
        window_type: ``'hann'`` or ``'hamming'``.

    Returns:
        (spectra, magnitudes):
          - spectra[n_frames][n_fft] — complex FFT of each frame.
          - magnitudes[n_frames][n_bins] — magnitude spectrum (pos. freqs).
    """
    window = (hamming_window(frame_size) if window_type == 'hamming'
              else hann_window(frame_size))

    spectra: list[list[complex]] = []
    mags: list[list[float]] = []

    n = len(samples)
    start = 0
    while start + frame_size <= n:
        frame = [samples[start + i] * window[i] for i in range(frame_size)]
        X = fft(frame)
        spectra.append(X)

        half = len(X) // 2
        mags.append([abs(X[i]) for i in range(half + 1)])

        start += hop_size

    return spectra, mags


# ═══════════════════════════════════════════════════════════════
# 5 — MEL SCALE  &  FILTERBANK
# ═══════════════════════════════════════════════════════════════

def hz_to_mel(freq: float) -> float:
    """Convert Hz to mel frequency scale.

    mel = 2595 · log10(1 + freq/700)
    """
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class MelFilterBank:
    """Triangular filterbank on the mel scale.

    Maps a linear-frequency magnitude spectrum to mel-frequency bands
    by applying a set of overlapping triangular filters whose center
    frequencies are equally spaced on the mel scale.
    """

    def __init__(self, n_filters: int = 26, n_fft: int = 1024,
                 sample_rate: int = 16000,
                 f_min: float = 100.0, f_max: float = 4000.0):
        """Initialise and build the filterbank.

        Args:
            n_filters: Number of mel filters.
            n_fft: FFT size (determines frequency resolution).
            sample_rate: Audio sample rate (Hz).
            f_min: Lowest frequency (Hz).
            f_max: Highest frequency (Hz).
        """
        self.n_filters = n_filters
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.filters: list[list[float]] = self._build()
        self.n_bins = n_fft // 2 + 1

    # ── Filterbank construction ────────────────────────────────

    def _build(self) -> list[list[float]]:
        """Return a list of ``n_filters`` triangular filter vectors."""
        n_filt = self.n_filters
        sr = self.sample_rate
        n_fft = self.n_fft

        # Mel-spaced centre frequencies (n_filters + 2 points for the
        # left/right boundaries of the first/last filter).
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_pts = [mel_min + i * (mel_max - mel_min) / (n_filt + 1)
                   for i in range(n_filt + 2)]
        hz_pts = [mel_to_hz(m) for m in mel_pts]

        # Map to FFT bin indices
        bin_pts = [int((n_fft + 1) * hz / sr) for hz in hz_pts]
        n_bins = n_fft // 2 + 1

        filters: list[list[float]] = []
        for m in range(1, n_filt + 1):
            left   = bin_pts[m - 1]
            centre = bin_pts[m]
            right  = bin_pts[m + 1]

            fbank = [0.0] * n_bins

            # Rising edge
            if centre > left:
                for k in range(left, centre):
                    fbank[k] = (k - left) / (centre - left)

            # Falling edge
            if right > centre:
                for k in range(centre, min(right, n_bins)):
                    fbank[k] = (right - k) / (right - centre)

            filters.append(fbank)

        return filters

    # ── Apply ──────────────────────────────────────────────────

    def apply(self, magnitude_spectrum: list[float]) -> list[float]:
        """Apply the filterbank to a magnitude spectrum.

        Args:
            magnitude_spectrum: Length must equal ``n_fft // 2 + 1``.

        Returns:
            Per-filter energies (log-compress externally for MFCC).
        """
        return [
            sum(m * f for m, f in zip(magnitude_spectrum, fbank))
            for fbank in self.filters
        ]

    def __repr__(self) -> str:
        return (f"MelFilterBank({self.n_filters} filters, "
                f"FFT={self.n_fft}, sr={self.sample_rate}, "
                f"[{self.f_min}-{self.f_max}] Hz)")


# ═══════════════════════════════════════════════════════════════
# 6 — DCT  (Type-II, used in MFCC)
# ═══════════════════════════════════════════════════════════════

def dct_ii(x: list[float]) -> list[float]:
    """Discrete Cosine Transform (Type-II).

    ``X[k] = sum_{n=0}^{N-1} x[n] · cos(π k (n+0.5) / N)``

    Args:
        x: Input sequence.

    Returns:
        DCT coefficients of the same length.
    """
    n = len(x)
    out = [0.0] * n
    for k in range(n):
        s = 0.0
        for t in range(n):
            s += x[t] * math.cos(math.pi * k * (t + 0.5) / n)
        out[k] = s
    return out


# ═══════════════════════════════════════════════════════════════
# 7 — MFCC
# ═══════════════════════════════════════════════════════════════

def compute_mfcc(samples: list[float], sample_rate: int = 16000,
                 n_mfcc: int = 13, n_fft: int = 1024,
                 hop_size: int = 512, n_filters: int = 26,
                 f_min: float = 100.0, f_max: float = 4000.0
                 ) -> tuple[list[list[float]], list[float]]:
    """Compute MFCC features from audio samples.

    Pipeline:
      1. STFT → magnitude spectrogram.
      2. Mel filterbank → mel-band energies.
      3. Log compression (with floor).
      4. DCT-II → MFCC coefficients.
      5. Keep first ``n_mfcc`` coefficients.

    Args:
        samples: Input audio.
        sample_rate: Sample rate (Hz).
        n_mfcc: Number of coefficients to return per frame.
        n_fft: FFT size.
        hop_size: STFT hop size.
        n_filters: Number of mel filters.
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).

    Returns:
        (mfcc_frames, time_stamps):
          - mfcc_frames[n_frames][n_mfcc]
          - time_stamps[n_frames]  in seconds.
    """
    # STFT magnitude spectrogram
    _, mag_frames = stft(samples, frame_size=n_fft, hop_size=hop_size)
    if not mag_frames:
        return [], []

    # Mel filterbank
    mfb = MelFilterBank(n_filters, n_fft, sample_rate, f_min, f_max)

    mfcc_frames: list[list[float]] = []
    time_stamps: list[float] = []

    for i, mag in enumerate(mag_frames):
        # Mel energies → log → DCT → keep first n_mfcc
        mel = mfb.apply(mag)
        log_mel = [math.log(max(e, 1e-10)) for e in mel]
        coeffs = dct_ii(log_mel)[:n_mfcc]
        mfcc_frames.append(coeffs)
        time_stamps.append(i * hop_size / sample_rate)

    return mfcc_frames, time_stamps


# ═══════════════════════════════════════════════════════════════
# 8 — VAD  (Voice Activity Detection)
# ═══════════════════════════════════════════════════════════════

class VAD:
    """Voice Activity Detection using energy + spectral flatness.

    A frame is classified as **voice** if:
      - its RMS energy exceeds *energy_threshold*, **and**
      - its spectral flatness is below *spectral_flatness_threshold*
        (i.e. the spectrum is tonal rather than noise-like).
    """

    def __init__(self, sample_rate: int = 16000,
                 frame_size: int = 256, hop_size: int = 128,
                 energy_threshold: float = 0.01,
                 spectral_flatness_threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.energy_threshold = energy_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold

    # ── Per-frame features ─────────────────────────────────────

    @staticmethod
    def rms_energy(frame: list[float]) -> float:
        """Root-mean-square energy of a frame."""
        if not frame:
            return 0.0
        return math.sqrt(sum(s * s for s in frame) / len(frame))

    @staticmethod
    def spectral_flatness(magnitudes: list[float]) -> float:
        """Spectral flatness (Wiener entropy).

        Flatness = geometric_mean / arithmetic_mean of the magnitude
        spectrum.  Close to 1.0 → noise-like; close to 0.0 → tonal.

        Args:
            magnitudes: Magnitude spectrum (positive frequencies).
        """
        mags = [m for m in magnitudes if m > 1e-12]
        if not mags:
            return 1.0

        n = len(mags)
        log_geom = sum(math.log(m) for m in mags) / n
        geom = math.exp(log_geom)
        arith = sum(mags) / n
        return geom / arith if arith > 1e-12 else 1.0

    # ── Decision ───────────────────────────────────────────────

    def is_voice_frame(self, frame: list[float]
                       ) -> tuple[bool, dict]:
        """Classify a single frame.

        Returns:
            (is_voice, info):
              - is_voice: ``True`` if voice activity detected.
              - info: dict with ``'energy'``, ``'spectral_flatness'``,
                ``'energy_voiced'``, ``'flatness_voiced'``.
        """
        energy = self.rms_energy(frame)
        X = fft(frame)
        mags = magnitude_spectrum(X)
        flatness = self.spectral_flatness(mags)

        energy_ok = energy > self.energy_threshold
        flatness_ok = flatness < self.spectral_flatness_threshold

        info = {
            'energy': energy,
            'spectral_flatness': flatness,
            'energy_voiced': energy_ok,
            'flatness_voiced': flatness_ok,
        }
        return (energy_ok and flatness_ok), info

    def process(self, samples: list[float]
                ) -> tuple[list[bool], list[float], list[dict]]:
        """Run VAD over the full signal.

        Returns:
            (decisions, time_stamps, infos):
              - decisions[n_frames] — bool per frame.
              - time_stamps[n_frames] — centre time in seconds.
              - infos[n_frames] — debug dicts.
        """
        decisions: list[bool] = []
        times: list[float] = []
        infos: list[dict] = []

        n = len(samples)
        start = 0
        while start + self.frame_size <= n:
            frame = samples[start:start + self.frame_size]
            is_voice, info = self.is_voice_frame(frame)
            decisions.append(is_voice)
            times.append(start / self.sample_rate)
            infos.append(info)
            start += self.hop_size

        return decisions, times, infos


# ═══════════════════════════════════════════════════════════════
# 9 — AUDIO CODEC  (Simple RVQ-like quantisation)
# ═══════════════════════════════════════════════════════════════

class AudioCodec:
    """Simple audio codec with residual vector quantisation (RVQ).

    Encodes audio frames into a small number of codebook indices
    across multiple stages.  Each stage quantises the residual of
    the previous stage — this is the same principle used in modern
    neural audio codecs (SoundStream, EnCodec).

    Codebooks are learned via online k-means on the training data.
    This is an *educational* implementation — not production grade.
    """

    def __init__(self, frame_size: int = 512,
                 n_codebooks: int = 4,
                 codebook_size: int = 16,
                 sample_rate: int = 16000):
        """
        Args:
            frame_size: Samples per frame.
            n_codebooks: Number of RVQ stages.
            codebook_size: Vectors per codebook.
            sample_rate: Audio sample rate.
        """
        self.frame_size = frame_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.codebooks: list[list[list[float]]] | None = None
        self.trained = False

    # ── K-means helper ─────────────────────────────────────────

    @staticmethod
    def _kmeans(vectors: list[list[float]], k: int,
                max_iter: int = 20) -> list[list[float]]:
        """Simple k-means clustering.

        Args:
            vectors: Data points (each is a list of floats).
            k: Number of clusters.
            max_iter: Iteration limit.

        Returns:
            ``k`` centroid vectors.
        """
        if not vectors:
            return []
        dim = len(vectors[0])
        n = len(vectors)

        # Initialise with random data points (no duplicates).
        centroids: list[list[float]] = []
        seen: set[int] = set()
        for _ in range(min(k, n)):
            while True:
                idx = random.randrange(n)
                if idx not in seen:
                    seen.add(idx)
                    centroids.append(list(vectors[idx]))
                    break
        # Pad with zeros if not enough points
        while len(centroids) < k:
            centroids.append([0.0] * dim)

        for _ in range(max_iter):
            # Assign
            assignments = []
            for vec in vectors:
                best = 0
                best_d = float('inf')
                for j, c in enumerate(centroids):
                    d = sum((vec[i] - c[i]) ** 2 for i in range(dim))
                    if d < best_d:
                        best_d = d
                        best = j
                assignments.append(best)

            # Update
            new_c = [[0.0] * dim for _ in range(k)]
            counts = [0] * k
            for idx, vec in zip(assignments, vectors):
                for i in range(dim):
                    new_c[idx][i] += vec[i]
                counts[idx] += 1
            for j in range(k):
                if counts[j] > 0:
                    for i in range(dim):
                        new_c[j][i] /= counts[j]
            centroids = new_c

        return centroids

    # ── Train ──────────────────────────────────────────────────

    def train(self, samples: list[float]) -> None:
        """Learn codebooks from audio samples.

        Frames are extracted, then each RVQ stage learns a codebook
        on the residuals from the previous stage.

        Args:
            samples: Training audio.
        """
        # Frame
        frames = []
        n = len(samples)
        start = 0
        while start + self.frame_size <= n:
            frames.append(samples[start:start + self.frame_size])
            start += self.frame_size
        if not frames:
            raise ValueError("Not enough samples for training "
                             f"(need at least {self.frame_size})")

        residuals: list[list[float]] = [list(f) for f in frames]
        self.codebooks = []

        for stage in range(self.n_codebooks):
            cb = self._kmeans(residuals, self.codebook_size)
            self.codebooks.append(cb)

            # Compute new residuals
            new_res: list[list[float]] = []
            for vec in residuals:
                # nearest centroid
                best_j = 0
                best_d = float('inf')
                for j, c in enumerate(cb):
                    d = sum((vec[i] - c[i]) ** 2 for i in range(self.frame_size))
                    if d < best_d:
                        best_d = d
                        best_j = j
                new_res.append([vec[i] - cb[best_j][i] for i in range(self.frame_size)])
            residuals = new_res

        self.trained = True

    # ── Encode ─────────────────────────────────────────────────

    def encode(self, samples: list[float]) -> list[list[int]]:
        """Encode audio into codebook indices.

        Args:
            samples: Audio signal.

        Returns:
            ``indices[n_frames][n_codebooks]`` — each entry is an index
            into the corresponding codebook.
        """
        if not self.trained:
            raise RuntimeError("Codec not trained")

        # Frame
        frames = []
        n = len(samples)
        start = 0
        while start + self.frame_size <= n:
            frames.append(samples[start:start + self.frame_size])
            start += self.frame_size

        all_idx: list[list[int]] = []
        for frame in frames:
            vec = list(frame)
            indices: list[int] = []
            for stage, cb in enumerate(self.codebooks):
                best_j = 0
                best_d = float('inf')
                for j, c in enumerate(cb):
                    d = sum((vec[i] - c[i]) ** 2 for i in range(self.frame_size))
                    if d < best_d:
                        best_d = d
                        best_j = j
                indices.append(best_j)
                vec = [vec[i] - cb[best_j][i] for i in range(self.frame_size)]
            all_idx.append(indices)
        return all_idx

    # ── Decode ─────────────────────────────────────────────────

    def decode(self, indices: list[list[int]]) -> list[float]:
        """Reconstruct audio from codebook indices.

        Args:
            indices: ``[n_frames][n_codebooks]`` — encoded representation.

        Returns:
            Decoded audio samples.
        """
        if not self.trained:
            raise RuntimeError("Codec not trained")

        out: list[float] = []
        for frame_idx in indices:
            frame = [0.0] * self.frame_size
            for stage, idx in enumerate(frame_idx):
                centroid = self.codebooks[stage][idx]
                for i in range(self.frame_size):
                    frame[i] += centroid[i]
            out.extend(frame)
        return out

    # ── Round-trip ─────────────────────────────────────────────

    def compress(self, samples: list[float]
                 ) -> tuple[list[float], list[list[int]]]:
        """Encode then decode (full round-trip).

        Returns:
            (decoded, encoded_indices).
        """
        encoded = self.encode(samples)
        decoded = self.decode(encoded)
        return decoded, encoded


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

def demo() -> None:
    """Run all audio components on synthetic data."""
    print("🎵 Anggira — Audio Module Demo")
    print("=" * 60)

    SR = 16000
    gen = AudioGenerator()

    # ── 1  AudioGenerator ──────────────────────────────────────
    print("\n── 1. AudioGenerator ──────────────────────")
    sine = gen.sine(440, 0.5, SR)
    sq   = gen.square(440, 0.5, SR)
    chirp_sig = gen.chirp(200, 2000, 1.0, SR)
    sil  = gen.silence(0.1, SR)
    print(f"  Sine 440 Hz × 0.5 s    → {len(sine):>5} samples")
    print(f"  Square 440 Hz × 0.5 s  → {len(sq):>5} samples")
    print(f"  Chirp 200→2000 Hz × 1s → {len(chirp_sig):>5} samples")
    print(f"  Silence 0.1 s          → {len(sil):>5} samples")
    print(f"  Sine[:5] = {[round(s, 4) for s in sine[:5]]}")

    # ── 2  FFT ─────────────────────────────────────────────────
    print("\n── 2. FFT (Cooley–Tukey) ───────────────────")
    t = [math.sin(TAU * 100 * i / 1024) for i in range(1024)]
    X = fft(t)
    mags = magnitude_spectrum(X)
    peak = max(range(len(mags)), key=lambda i: mags[i])
    print(f"  Signal: 100 Hz sine, 1024 samples")
    print(f"  FFT bins: {len(X)}")
    print(f"  Peak magnitude  @ bin {peak}  →  {mags[peak]:.2f}")

    # Verify against DFT
    Xd = dft(t)
    diff = sum(abs(X[i] - Xd[i]) for i in range(len(Xd)))
    print(f"  FFT ≈ DFT  diff:  {diff:.2e}")

    # FFT → IFFT round-trip
    recovered = ifft(X)
    rt_err = sum(abs(t[i] - recovered[i].real) for i in range(1024)) / 1024
    print(f"  FFT→IFFT error:      {rt_err:.2e}")

    # Non-power-of-2
    t197 = t[:197]
    X197 = fft(t197)
    print(f"  FFT(len=197) → padded to {len(X197)} bins")

    # ── 3  STFT ────────────────────────────────────────────────
    print("\n── 3. STFT ──────────────────────────────────")
    audio_stft = gen.chirp(200, 3000, 0.5, SR)
    spec, mag_spec = stft(audio_stft, frame_size=256, hop_size=128)
    print(f"  Input:  {len(audio_stft)} samples")
    print(f"  Frames: {len(spec)}")
    print(f"  Bins/frame: {len(mag_spec[0])}")
    print(f"  Spectrogram: {len(mag_spec)} frames × {len(mag_spec[0])} bins")
    print(f"  Frame 0 magnitude[:5] = {[round(m, 2) for m in mag_spec[0][:5]]}")

    # ── 4  MelFilterBank ───────────────────────────────────────
    print("\n── 4. MelFilterBank ─────────────────────────")
    mfb = MelFilterBank(n_filters=13, n_fft=256, sample_rate=SR)
    mel_e = mfb.apply(mag_spec[0])
    print(f"  {mfb}")
    print(f"  Frame 0 mel energies: {[round(e, 4) for e in mel_e]}")

    # ── 5  MFCC ────────────────────────────────────────────────
    print("\n── 5. MFCC ──────────────────────────────────")
    audio_mfcc = gen.chirp(200, 3000, 0.5, SR)
    mfccs, times = compute_mfcc(audio_mfcc, sample_rate=SR,
                                n_mfcc=13, n_filters=26)
    print(f"  Frames: {len(mfccs)}")
    print(f"  Coefficients/frame: {len(mfccs[0])}")
    print(f"  Frame 0 → {[round(c, 4) for c in mfccs[0]]}")
    if len(mfccs) > 1:
        print(f"  Frame 1 → {[round(c, 4) for c in mfccs[1]]}")

    # ── 6  VAD ─────────────────────────────────────────────────
    print("\n── 6. VAD ──────────────────────────────────")
    voiced   = gen.sine(300, 0.3, SR)          # tonal, high energy
    unvoiced = gen.chirp(200, 2000, 0.3, SR)   # noise-like (sweep)
    silent   = gen.silence(0.2, SR)
    mixed    = voiced + unvoiced + silent

    vad = VAD(sample_rate=SR, energy_threshold=0.008,
              spectral_flatness_threshold=0.5)
    dec, vtimes, infos = vad.process(mixed)
    n_voice = sum(dec)
    print(f"  Mixed audio: {len(mixed)} samples")
    print(f"  Total frames: {len(dec)}, Voice frames: {n_voice}")
    for i in range(min(6, len(dec))):
        info = infos[i]
        print(f"    frame {i} @ {vtimes[i]:.3f}s  "
              f"voice={dec[i]}  "
              f"E={info['energy']:.4f}  "
              f"SF={info['spectral_flatness']:.4f}")

    # ── 7  AudioCodec ──────────────────────────────────────────
    print("\n── 7. AudioCodec (RVQ) ──────────────────────")
    train_sig = gen.sine(440, 0.5, SR) + gen.chirp(200, 1000, 0.5, SR)
    test_sig  = gen.sine(550, 0.3, SR)

    codec = AudioCodec(frame_size=128, n_codebooks=3, codebook_size=8)
    print("  Training codebooks … ", end='', flush=True)
    codec.train(train_sig)
    print("done.")
    print(f"  Codebooks: {codec.n_codebooks} stages × "
          f"{codec.codebook_size} vectors "
          f"(dim={codec.frame_size})")

    decoded, encoded = codec.compress(test_sig)
    if decoded:
        min_len = min(len(test_sig), len(decoded))
        sig_pwr = sum(s ** 2 for s in test_sig[:min_len])
        noi_pwr = sum((test_sig[i] - decoded[i]) ** 2
                      for i in range(min_len))
        snr = (10 * math.log10(sig_pwr / noi_pwr)
               if noi_pwr > 1e-12 else float('inf'))
        print(f"  Encoded: {len(encoded)} frames × {len(encoded[0])} indices")
        print(f"  Decoded: {len(decoded)} samples")
        print(f"  SNR:     {snr:.2f} dB")
        print(f"  First frame indices: {encoded[0]}")
    else:
        print("  (test signal too short for one frame)")

    # ── Done ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Anggira audio module OK — all components working.")


if __name__ == '__main__':
    demo()
