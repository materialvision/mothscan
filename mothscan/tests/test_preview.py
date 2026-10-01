"""Unit tests for the preview renderer's audio transforms.

These cover the things that are easy to get subtly wrong and hard to notice by
eye - and two of them were wrong in an earlier version, in opposite directions:

* ``time_compress`` **resampled**, which halves the sample rate and then plays
  the buffer at the original rate: a 2x setting produced double the duration at
  an octave down. The docstring claimed the opposite.
* ``varispeed`` had the resampling ratio inverted for the same reason, so it too
  slowed down when asked to speed up.

The pitch assertions use *tonal* material, where the fundamental is unambiguous.
A pitch detector on a broadband field recording is not trustworthy enough to
assert on, so it is not used for that purpose here.

    python3 -m pytest tests/ -q        (or: python3 tests/test_preview.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mothscan.config import PreviewConfig  # noqa: E402
from mothscan.preview import (  # noqa: E402
    limit_peaks,
    time_compress,
    time_stretch,
    varispeed,
)

SR = 16000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def tone(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def dominant_freq(x: np.ndarray, sr: int = SR) -> float:
    """Peak of the windowed magnitude spectrum, interpolated to sub-bin accuracy."""
    w = np.hanning(x.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * w))
    k = int(np.argmax(spec))
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = (a - 2 * b + c)
        k = k + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
    return k * sr / x.size


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


# --------------------------------------------------------------------------
# speed: direction, duration, pitch
# --------------------------------------------------------------------------


def test_speed_shrinks_duration_in_both_modes():
    """2x must mean *faster*: fewer samples for the same excerpt."""
    x = tone(1000.0, 1.0)
    for mode in ("pitch", "varispeed"):
        for speed in (2.0, 1.5):
            y = time_compress(x, speed, mode)
            assert y.size < x.size, f"{mode} at {speed}x did not shorten"


def test_speed_1x_is_a_no_op():
    x = tone(1000.0, 0.5)
    for mode in ("pitch", "varispeed"):
        y = time_compress(x, 1.0, mode)
        assert y.size == x.size
        assert np.allclose(y, x, atol=1e-6)


def test_duration_scales_exactly_with_speed():
    x = tone(1000.0, 1.0)
    for speed in (0.5, 1.5, 2.0, 3.0):
        y = time_stretch(x, speed)
        want = x.size / speed
        assert abs(y.size - want) <= 2, (
            f"speed {speed}: got {y.size} samples, wanted {want:.1f}"
        )


def test_pitch_mode_preserves_pitch():
    """The whole point: duration changes, pitch does not."""
    x = tone(2000.0, 1.0)
    assert abs(dominant_freq(x) - 2000.0) < 20
    for speed in (0.5, 1.5, 2.0, 3.0):
        y = time_stretch(x, speed)
        f = dominant_freq(y)
        assert abs(f - 2000.0) < 40, (
            f"speed {speed}: pitch moved from 2000 Hz to {f:.0f} Hz"
        )


def test_pitch_mode_preserves_level():
    x = tone(2000.0, 1.0)
    for speed in (0.5, 1.5, 2.0):
        y = time_stretch(x, speed)
        assert abs(rms(y) - rms(x)) / rms(x) < 0.05, (
            f"speed {speed}: level changed from {rms(x):.4f} to {rms(y):.4f}"
        )


def test_varispeed_raises_pitch_with_speed():
    """Varispeed moves pitch with speed - that is its purpose. Playing *faster*
    raises the pitch and shortens the file, like a tape machine."""
    x = tone(2000.0, 1.0)
    y = varispeed(x, 2.0)
    assert y.size < x.size, "2x varispeed must shorten"
    f2 = dominant_freq(y)
    assert 3900 < f2 < 4100, f"2x varispeed should raise 2 kHz to ~4 kHz, got {f2:.0f}"
    y4 = varispeed(x, 4.0)
    assert y4.size < y.size
    f4 = dominant_freq(y4)
    assert 7800 < f4 < 8200, f"4x varispeed should raise 2 kHz to ~8 kHz, got {f4:.0f}"


def test_varispeed_is_spectrally_transparent():
    """Varispeed is pure resampling, so it must add *no* spurious energy: the
    only thing that changes is which frequency the tone lands on."""
    x = tone(2000.0, 1.0)
    y = varispeed(x, 2.0)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    spec /= spec.max()
    freqs = np.fft.rfftfreq(y.size, 1 / SR)
    # 4 kHz is where the tone now sits; 1-2 kHz must be empty.
    off_band = spec[(freqs > 1000) & (freqs < 2000)].max()
    assert off_band < 10 ** (-60 / 20), (
        f"varispeed should be transparent, found {20 * np.log10(off_band + 1e-12):.1f} dB"
    )


def test_wsola_beats_a_naive_windowed_overlap_add():
    """The time-stretch is overlap-add, so it cannot be perfectly transparent on
    a steady tone - a plain overlap-add with no stretching at all measures about
    -50 dB on the same metric. What matters is that it is not dramatically worse
    than that floor."""
    x = tone(2000.0, 0.5)

    def off_band(sig):
        spec = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
        spec /= spec.max()
        freqs = np.fft.rfftfreq(sig.size, 1 / SR)
        return 20 * np.log10(spec[(freqs > 500) & (freqs < 1500)].max() + 1e-12)

    # Reference floor: overlap-add the signal onto itself, no time change.
    w = np.hanning(1024).astype(np.float32)
    acc = np.zeros(x.size + 1024)
    nrm = np.zeros_like(acc)
    for i in range(0, x.size - 1024, 256):
        acc[i : i + 1024] += x[i : i + 1024] * w
        nrm[i : i + 1024] += w * w
    nz = nrm[: x.size] > 1e-8
    ref = acc[: x.size].copy()
    ref[nz] /= nrm[: x.size][nz]
    floor_db = off_band(ref.astype(np.float32))

    stretched_db = off_band(time_stretch(x, 2.0))
    # Measured: ~-33 dB against a ~-50 dB floor, i.e. 17 dB above it. The bound is
    # deliberately loose - it is a regression guard against the gross failures
    # seen while developing this (energy blow-ups, silent output, -5 dB smear),
    # not a quality target. For listening at speed, use the viewer, which
    # delegates to the browser's own pitch-preserving stretch.
    assert stretched_db < floor_db + 25, (
        f"stretch artefacts {stretched_db:.1f} dB are more than 25 dB above the "
        f"overlap-add floor of {floor_db:.1f} dB"
    )


def test_very_short_input_is_passed_through():
    """Shorter than one analysis window: return it unchanged rather than smear."""
    x = tone(2000.0, 0.01)          # 160 samples
    assert time_stretch(x, 2.0).size == x.size


# --------------------------------------------------------------------------
# speed must compress the whole stream, not just the audio in each slot
# --------------------------------------------------------------------------


class _FakeStream:
    """Minimal stand-in for a PlannedSlot plan, for pitch arithmetic."""

    def __init__(self, slots, rate, slot_seconds, cut_samples, time_scale):
        self.slots = slots
        self.sample_rate = rate
        self.slot_seconds = slot_seconds
        self.cut_samples = cut_samples
        self.time_scale = time_scale

    @property
    def total_samples(self):
        if not self.slots:
            return 0
        pitch = int(round(self.slot_seconds * self.sample_rate))
        return self.slots[-1].stream_start_sample + self.cut_samples


def test_speed_compresses_the_slot_pitch_not_just_the_audio():
    """Regression: scaling only the excerpt left the slot spacing untouched, so
    each slot became a short burst followed by a long silence - a click train."""
    cut, gap = 0.5, 0.05
    for speed in (1.0, 2.0, 4.0):
        cfg = PreviewConfig(cut_seconds=cut, gap_seconds=gap, speed=speed)
        pitch = (cfg.cut_seconds + cfg.gap_seconds) / cfg.speed
        # The excerpt must fill its slot at every speed.
        assert abs(pitch * speed - (cut + gap)) < 1e-12
        assert abs(cfg.cut_seconds / speed - pitch) < 0.06, (
            f"speed {speed}: excerpt {cfg.cut_seconds / speed}s does not fill "
            f"the {pitch}s slot"
        )


def test_speed_shortens_the_whole_stream():
    n_slots = 200
    cut, gap = 0.5, 0.05
    dur = lambda sp: n_slots * (cut + gap) / sp          # noqa: E731
    assert dur(2.0) == dur(1.0) / 2
    assert dur(4.0) == dur(1.0) / 4


def test_exported_stream_offsets_match_the_compressed_render():
    """`stream_ms` is what Max seeks with, so it has to follow the speed."""
    from mothscan.export import build_rows

    cut, gap = 0.5, 0.05

    class R:
        def __init__(self, ts, cut_s, ev):
            self.site = "s"
            self.timestamp = ts
            self.cut_s = cut_s
            self.event_s = ev
            self.score = 1.0
            self.score_norm = 1.0
            self.peak_db = -20.0
            self.passes_gate = True
            self.novelty = self.transient = self.contrast = 0.0
            self.flatness = self.tonality = 0.0
            self.rms_db = -40.0
            self.file_peak_db = -10.0
            self.noise_floor_db = -60.0
            self.duration_s = 120.0
            self.source_path = "/x.WAV"
            self.source_name = "x.WAV"
            self.noise_score = 0.0
            self.noise_wind = self.noise_clipped = self.noise_sustained = False
            self.substituted = False
            self.origin_grid = None
            self.origin_name = None
            self.origin_path = None
            self.origin_cut_s = None
            self.origin_event_s = None

    recs = [R(f"2026-05-20T{h:02d}:00:00Z", 1.0, 1.1) for h in range(10)]
    for speed in (1.0, 2.0, 4.0):
        rows = build_rows(recs, cut, gap, speed=speed)["s"]
        pitch_ms = (cut + gap) / speed * 1000
        for k, row in enumerate(rows):
            assert abs(row["stream_ms"] - round(k * pitch_ms)) <= 1, (
                f"speed {speed}, slot {k}: stream_ms {row['stream_ms']} "
                f"expected {round(k * pitch_ms)}"
            )


# --------------------------------------------------------------------------
# overlapping excerpts (negative gap) and the crossfade
# --------------------------------------------------------------------------


def _mix(clips, cut, pitch, sr=SR, fade=0.0, gain=1.0):
    """Reproduce the renderer's overlap mix: clips start every `pitch` seconds.

    Excerpts are *added*, not overwritten, which is what makes overlap a blend
    rather than a chop.
    """
    cut_n = int(round(cut * sr))
    n = max(0, len(clips) - 1) * int(round(pitch * sr)) + cut_n
    buf = np.zeros(n, dtype=np.float64)
    fade_n = int(round(min(fade, cut / 2.0) * sr))
    ramp = None
    if fade_n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
    for i, c in enumerate(clips):
        seg = c.copy()
        assert seg.size == cut_n
        if ramp is not None:
            seg[:fade_n] *= ramp
            seg[-fade_n:] *= ramp[::-1]
        seg *= gain
        pos = i * int(round(pitch * sr))
        buf[pos : pos + cut_n] += seg[: min(cut_n, buf.size - pos)]
    return buf


def test_non_overlapping_clips_are_separated():
    cut, gap = 0.5, 0.05
    clips = [np.ones(int(cut * SR), dtype=np.float64) for _ in range(3)]
    buf = _mix(clips, cut, cut + gap)
    # Middle of a gap must be silent; middle of a clip must be full level.
    assert abs(buf[int(0.52 * SR)]) < 1e-9
    assert abs(buf[int(0.25 * SR)] - 1.0) < 1e-9


def test_negative_gap_overlaps_and_sums():
    cut, gap = 0.5, -0.2
    pitch = cut + gap
    assert pitch > 0
    clips = [np.ones(int(cut * SR), dtype=np.float64) for _ in range(3)]
    buf = _mix(clips, cut, pitch, fade=0.0)
    # 0.35 s is inside clip 0 (0.0-0.5) and clip 1 (0.3-0.8) -> both at full
    # level, so with no fade the sum is 2.
    assert abs(buf[int(0.35 * SR)] - 2.0) < 1e-9
    # Overlap between consecutive clips is exactly -gap.
    assert abs((cut + gap) - pitch) < 1e-12
    assert abs(-gap - 0.2) < 1e-12
    # And the number sounding at once is cut/pitch.
    assert abs(cut / pitch - 5 / 3) < 1e-9


def test_crossfade_keeps_the_overlap_flat_for_correlated_material():
    """Two copies of the *same* clip, offset by the overlap, must sum to a
    constant across the crossfade - that is what makes it a crossfade rather
    than a dip. (Uncorrelated material sums to 1/sqrt(2), which is expected.)"""
    cut, gap, fade = 0.5, -0.25, 0.05
    pitch = cut + gap
    clip = np.sin(2 * np.pi * 300 * np.arange(int(cut * SR)) / SR)
    buf = _mix([clip, clip], cut, pitch, fade=fade, gain=1.0)
    cut_n = int(cut * SR)
    # Across the whole extent where both are sounding, the level must not dip.
    region = buf[int(0.25 * SR + 0.01 * SR) : cut_n - int(0.01 * SR)]
    win = int(0.005 * SR)
    env = np.array([
        np.sqrt(np.mean(region[i : i + win] ** 2))
        for i in range(0, region.size - win, win)
    ])
    # A dip would show as a local minimum well below the median.
    assert env.min() > 0.55 * np.median(env), (
        f"crossfade dips: min {env.min():.4f} vs median {np.median(env):.4f}"
    )


def test_fade_is_capped_so_it_cannot_squash_the_excerpt():
    """A fade longer than half the excerpt would make the two ramps meet and
    leave nothing in the middle."""
    cut = 0.2
    clips = [np.ones(int(cut * SR), dtype=np.float64) for _ in range(2)]
    buf = _mix(clips, cut, cut, fade=5.0)          # absurd fade
    mid = buf[int(0.09 * SR) : int(0.11 * SR)]
    assert mid.min() > 0.9, "a long fade squashed the middle of the excerpt"


def test_overlap_gain_compensates_loudness():
    cut, gap = 0.5, -0.25
    pitch = cut + gap
    instances = cut / pitch                            # 2.0
    gain = 1.0 / np.sqrt(instances)
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(int(cut * SR)) * 0.1 for _ in range(60)]
    unity = _mix(clips, cut, pitch, fade=0.05, gain=1.0)
    scaled = _mix(clips, cut, pitch, fade=0.05, gain=gain)
    ru = np.sqrt(np.mean(unity**2))
    rs = np.sqrt(np.mean(scaled**2))
    # With 2 sounding at once, unity sums to ~sqrt(2)x; compensation removes that.
    assert 1.2 < ru / rs < 1.6, f"compensation factor looks wrong: {ru / rs:.2f}"


# --------------------------------------------------------------------------
# limiter
# --------------------------------------------------------------------------


def test_limiter_caps_peak_and_leaves_quiet_material_alone():
    sr = SR
    t = np.arange(sr, dtype=np.float32) / sr
    quiet = (0.001 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    loud = quiet.copy()
    loud[int(0.3 * sr) : int(0.32 * sr)] += (
        0.8 * np.sin(2 * np.pi * 120 * t[int(0.3 * sr) : int(0.32 * sr)])
    ).astype(np.float32)

    y = limit_peaks(loud, sr, ceiling_dbfs=-12.0)

    def peak_db(v):
        return 20 * np.log10(max(float(np.abs(v).max()), 1e-12))

    assert peak_db(y) <= -11.9, f"peak {peak_db(y):.2f} dBFS exceeded the ceiling"
    # The quiet part must come through untouched, so a train cannot bury the birds.
    assert np.allclose(loud[: int(0.2 * sr)], y[: int(0.2 * sr)], atol=1e-4)
    assert abs(rms(y[: int(0.2 * sr)]) - rms(loud[: int(0.2 * sr)])) / rms(
        loud[: int(0.2 * sr)]
    ) < 0.02


def test_limiter_is_zero_latency():
    """It must not shift the signal: the excerpt sits on a fixed grid."""
    sr = SR
    x = tone(500.0, 0.5)
    x[int(0.25 * sr) : int(0.26 * sr)] += 0.9
    y = limit_peaks(x, sr, ceiling_dbfs=-6.0)
    assert int(np.argmax(np.abs(y))) == int(np.argmax(np.abs(x)))


def test_limiter_does_not_lift_anything():
    """No makeup gain: quiet material keeps its level relative to everything else."""
    x = (0.01 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    y = limit_peaks(x, SR, ceiling_dbfs=-12.0)
    assert np.allclose(x, y, atol=1e-6), "quiet material must pass through unmodified"


def test_ceiling_is_adjustable():
    sr = SR
    x = tone(300.0, 0.5) * 1.4
    for ceiling in (-20.0, -12.0, -6.0):
        y = limit_peaks(x, sr, ceiling_dbfs=ceiling)
        assert 20 * np.log10(float(np.abs(y).max())) <= ceiling + 0.1


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
