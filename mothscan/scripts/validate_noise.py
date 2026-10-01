"""Visual validation of the wind / clipping classifier.

Renders a labelled gallery: the files the classifier calls wind, the files it
calls clipped, and the files it calls clean, side by side on the same dB scale.
Thresholds are only trustworthy once the three groups look obviously different.

    python3 scripts/validate_noise.py --per-group 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mothscan.config import DATA_DIR  # noqa: E402
from mothscan.features import load_mono  # noqa: E402
from mothscan.index_store import load_feature_cache  # noqa: E402
from mothscan.noise import ContaminationConfig, classify  # noqa: E402
from mothscan.pipeline import discover  # noqa: E402


def spectrogram(x: np.ndarray, sr: int, fmax: float = 6000.0, nfft: int = 2048):
    hop = nfft // 4
    win = np.hanning(nfft).astype(np.float32)
    nf = 1 + (len(x) - nfft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(nf, nfft), strides=(x.strides[0] * hop, x.strides[0])
    )
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    keep = f <= fmax
    db = 20 * np.log10(np.maximum(spec[:, keep], 1e-9)).T
    return db, f[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--out", default=str(DATA_DIR / "noise_validation"))
    ap.add_argument("--fmax", type=float, default=6000.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ContaminationConfig()
    groups: dict[str, list] = {"wind": [], "clipped": [], "clean": []}
    for site, path in discover():
        fe = load_feature_cache(path, site)
        if fe is None:
            continue
        c = classify(fe)
        rec = (site, path, fe, c)
        if c.wind:
            groups["wind"].append(rec)
        elif c.clipped:
            groups["clipped"].append(rec)
        else:
            groups["clean"].append(rec)

    print(f"populations: wind={len(groups['wind'])} clipped={len(groups['clipped'])} "
          f"clean={len(groups['clean'])}")

    # Pick a representative spread rather than the most extreme: the extremes are
    # easy to classify, the interesting cases are the borderline ones.
    def pick(pool, n):
        if not pool:
            return []
        pool = sorted(pool, key=lambda r: -classify(r[2], cfg).score)
        if len(pool) <= n:
            return pool
        idx = np.linspace(0, len(pool) - 1, n).astype(int)
        return [pool[i] for i in idx]

    for group, n in (("wind", args.per_group), ("clipped", args.per_group),
                     ("clean", args.per_group)):
        recs = pick(groups[group], n)
        if not recs:
            continue
        fig, axes = plt.subplots(len(recs), 2, figsize=(17, 2.05 * len(recs)),
                                 gridspec_kw={"width_ratios": [3, 1.15]})
        if len(recs) == 1:
            axes = axes[None, :]
        for row, (site, path, fe, c) in enumerate(recs):
            x, sr = load_mono(path)
            db, f = spectrogram(x, sr, args.fmax)
            ax = axes[row, 0]
            im = ax.pcolormesh(np.arange(db.shape[1]) * (2048 // 4) / sr, f, db,
                               shading="auto", cmap="magma",
                               vmin=np.percentile(db, 45), vmax=np.percentile(db, 99.9))
            ax.set_ylim(0, 700 if group != "clean" else args.fmax)
            ax.set_ylabel("Hz", fontsize=8)
            ax.set_title(
                f"{group.upper()}  {site} {path.name}   score={c.score:.2f}  "
                f"wind={c.wind} clip={c.clipped}   low100={100*fe.low100:.0f}%  "
                f"low40={100*fe.ulf_ratio:.1f}%  tilt={fe.tilt_db_oct:.1f} dB/oct  "
                f"crest={fe.crest:.1f}  peak={fe.peak_db:.1f} dBFS  "
                f"run={fe.clip_run} runs={fe.clip_runs}",
                fontsize=8, loc="left")
            if row == len(recs) - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=6)

            bx = axes[row, 1]
            bx.plot(np.arange(len(x)) / sr, x, lw=0.25, color="#94a3b8")
            bx.set_xlim(0, len(x) / sr)
            bx.tick_params(labelsize=6)
            if row == len(recs) - 1:
                bx.set_xlabel("time (s)", fontsize=8)
        fig.tight_layout()
        p = out / f"gallery_{group}.png"
        fig.savefig(p, dpi=72)
        plt.close(fig)
        print(f"  {p}")

    import json
    (out / "populations.json").write_text(json.dumps(
        {k: [{"site": s, "file": p.name, "score": classify(fe, cfg).score}
             for s, p, fe, _ in v] for k, v in groups.items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
