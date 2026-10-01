"""Calibrate the wind / clipping thresholds against the whole deployment.

Thresholds must be *earned*, not guessed, so this reads every cached feature set
and looks at the distributions of the contamination indicators, then reports the
flags a candidate threshold would raise and renders the worst offenders as
spectrograms to check by eye.

    python3 scripts/calibrate_noise.py                 # distributions + flags
    python3 scripts/calibrate_noise.py --flag          # + write flags into index
    python3 scripts/calibrate_noise.py --plots 12      # + spectrograms of worst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mothscan.config import DATA_DIR, SITE_ROOT, is_audio_file  # noqa: E402
from mothscan.features import load_mono  # noqa: E402
from mothscan.index_store import load_feature_cache  # noqa: E402
from mothscan.noise import ContaminationConfig, classify, indicators_row  # noqa: E402
from mothscan.pipeline import discover  # noqa: E402


def collect() -> list[dict]:
    rows = []
    for site, path in discover():
        feats = load_feature_cache(path, site)
        if feats is None:
            continue
        rows.append(
            {
                "site": site,
                "name": path.name,
                "path": str(path),
                "ulf_ratio": feats.ulf_ratio,
                "low100": feats.low100,
                "low200": feats.low200,
                "ulf_rel_db": feats.ulf_rel_db,
                "tilt_db_oct": feats.tilt_db_oct,
                "sustain": feats.sustain,
                "crest": feats.crest,
                "clip_run": feats.clip_run,
                "clip_runs": feats.clip_runs,
                "clip_frac": feats.clip_frac,
                "peak_db": feats.peak_db,
                "noise_floor_db": feats.noise_floor_db,
            }
        )
    return rows


def pct_table(rows: list[dict], keys: list[str]) -> None:
    print(f"{'indicator':14s} " + " ".join(f"{p:>9}" for p in
          ("p1", "p25", "p50", "p75", "p90", "p99", "max")))
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        q = np.percentile(v, [1, 25, 50, 75, 90, 99])
        print(f"{k:14s} " + " ".join(f"{x:9.4f}" for x in q) + f" {v.max():9.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", action="store_true", help="write flags into the index")
    ap.add_argument("--plots", type=int, default=0, help="spectrograms of worst N")
    ap.add_argument("--out", default=str(DATA_DIR / "noise_calibration"))
    args = ap.parse_args()

    rows = collect()
    print(f"cached feature sets: {len(rows)}\n")

    keys = ["low100", "low200", "ulf_ratio", "ulf_rel_db", "tilt_db_oct",
            "sustain", "crest", "clip_run", "clip_runs", "clip_frac"]
    pct_table(rows, keys)

    cfg = ContaminationConfig()
    print(f"\nthresholds in use:\n  {json.dumps(cfg.__dict__, indent=2)}")

    classes = [classify(indicators_row(r), cfg) for r in rows]
    from collections import Counter

    wind = sum(1 for c in classes if c.wind)
    clipped = sum(1 for c in classes if c.clipped)
    sust = sum(1 for c in classes if c.sustained)
    bad = sum(1 for c in classes if c.unusable)
    print(f"\nany rail touch: {clipped}   wind-dominated: {wind}   "
          f"sustained clipping: {sust}")
    print(f"unusable as a source (wind or sustained clip) = {bad} of {len(rows)} "
          f"({100 * bad / max(1, len(rows)):.2f}%)")

    per_site = Counter()
    for r, c in zip(rows, classes):
        if c.unusable:
            per_site[r["site"]] += 1
    for site, n in sorted(per_site.items()):
        tot = sum(1 for r in rows if r["site"] == site)
        print(f"   {site:14s} {n:5d} / {tot:5d}  ({100 * n / max(1, tot):.2f}%)")

    # Worst offenders by contamination score.
    order = np.argsort([-c.score for c in classes])
    print(f"\n{'score':>6} {'wind':>5} {'clip':>5} {'sust':>5} {'low100':>7} "
          f"{'tilt':>7} {'crest':>6} {'run':>5} {'runs':>6}  file")
    for i in order[: min(25, len(order))]:
        r, c = rows[i], classes[i]
        print(f"{c.score:6.2f} {str(c.wind):>5} {str(c.clipped):>5} "
              f"{str(c.sustained):>5} {100*r['low100']:6.1f}% "
              f"{r['tilt_db_oct']:7.2f} {r['crest']:6.1f} {int(r['clip_run']):5d} "
              f"{int(r['clip_runs']):6d}  {r['site'][:10]:10s} {r['name']}")

    if args.plots:
        render_plots(rows, order[: args.plots], Path(args.out))

    if args.flag:
        from mothscan.index_store import IndexStore

        with IndexStore() as store:
            n = 0
            for r, c in zip(rows, classes):
                store.conn.execute(
                    """UPDATE points SET ulf_ratio=?, tilt_db_oct=?, sustain=?,
                       crest=?, clip_run=?, clip_runs=?, clip_frac=?,
                       noise_score=?, noise_wind=?, noise_clipped=?
                       WHERE site=? AND source_name=?""",
                    (r["ulf_ratio"], r["tilt_db_oct"], r["sustain"], r["crest"],
                     r["clip_run"], r["clip_runs"], r["clip_frac"],
                     c.score, int(c.wind), int(c.clipped), r["site"], r["name"]),
                )
                n += 1
            store.conn.commit()
        print(f"\nwrote contamination flags for {n} files into the index")
    return 0


def render_plots(rows: list[dict], order, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    print(f"\nrendering {len(order)} spectrograms -> {out}")
    for rank, i in enumerate(order):
        r = rows[i]
        path = Path(r["path"])
        feats = load_feature_cache(path, r["site"])
        x, sr = load_mono(path)
        # Wide view of the whole 2 minutes, focused on the band wind lives in.
        n = 1024
        hop = 256
        win = np.hanning(n).astype(np.float32)
        nf = 1 + (len(x) - n) // hop
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(nf, n), strides=(x.strides[0] * hop, x.strides[0]))
        spec = np.abs(np.fft.rfft(frames * win, axis=1))
        f = np.fft.rfftfreq(n, 1.0 / sr)
        band = f <= 5000
        db = 20 * np.log10(np.maximum(spec[:, band], 1e-9)).T
        del spec, frames

        fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        t = np.arange(db.shape[1]) * hop / sr
        im = ax[0].pcolormesh(t, f[band], db, shading="auto", cmap="magma",
                              vmin=np.percentile(db, 40), vmax=np.percentile(db, 99.9))
        ax[0].set_ylabel("Hz")
        ax[0].set_title(
            f"rank {rank + 1}  {r['site']} {r['name']}   "
            f"score={classify(indicators_row(r)).score:.2f}  "
            f"ulf={100 * r['ulf_ratio']:.1f}%  tilt={r['tilt_db_oct']:.1f} dB/oct  "
            f"sustain={r['sustain']:.2f}  clip_run={r['clip_run']}"
        )
        fig.colorbar(im, ax=ax[0], pad=0.01, label="dB")
        ax[1].plot(np.arange(len(x)) / sr, x, lw=0.3, color="#94a3b8")
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("amp")
        ax[1].set_xlim(0, len(x) / sr)
        fig.tight_layout()
        fig.savefig(out / f"worst_{rank + 1:02d}_{r['name']}.png", dpi=80)
        plt.close(fig)
    print("done")


if __name__ == "__main__":
    raise SystemExit(main())
