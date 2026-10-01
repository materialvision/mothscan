"""Benchmark + audit the scorer over a sample of files from every site.

Run:  python3 scripts/bench.py [--per-site 4] [--site "moth oculus"]
Prints per-file runtime, the chosen moment, and the score breakdown so the
model can be judged (and tuned) before committing to the full 64 GB run.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mothscan.config import SITE_ROOT, AnalysisConfig, is_audio_file  # noqa: E402
from mothscan.features import extract_features  # noqa: E402
from mothscan.scoring import EventScorer, explain  # noqa: E402


def sample_files(site: str, count: int, seed: int = 7) -> list[Path]:
    d = SITE_ROOT / site
    files = sorted(p for p in d.iterdir() if p.is_file() and is_audio_file(p))
    if not files:
        return []
    rng = random.Random(seed)
    # Spread the sample across the whole date range rather than the first N.
    idx = np.linspace(0, len(files) - 1, min(count, len(files))).astype(int)
    picked = [files[i] for i in idx]
    rng.shuffle(picked)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-site", type=int, default=4)
    ap.add_argument("--site", default=None, help="single site to test")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = AnalysisConfig()
    sites = [args.site] if args.site else ["moth oculus", "moth-juvet", "moth-kvarvet"]

    total_t = 0.0
    n = 0
    print(f"AnalysisConfig: n_fft={cfg.n_fft} hop={cfg.hop} n_mels={cfg.n_mels} "
          f"scan={cfg.scan_seconds}s weights(nov,trn,con,flat,ton)="
          f"({cfg.w_novelty},{cfg.w_transient},{cfg.w_contrast},{cfg.w_flatness},{cfg.w_tonality})")
    print("=" * 108)

    for site in sites:
        files = sample_files(site, args.per_site, args.seed)
        print(f"\n### {site}  ({len(files)} sampled)")
        for f in files:
            t0 = time.perf_counter()
            feats = extract_features(f, cfg)
            t1 = time.perf_counter()
            scorer = EventScorer(feats, cfg)
            best, alts = scorer.find_best()
            t2 = time.perf_counter()

            total_t += t2 - t0
            n += 1
            rt = t2 - t0
            print(
                f"  {f.name}  audio={feats.duration_s:6.1f}s  feat={t1-t0:5.2f}s  "
                f"score={t2-t1:5.2f}s  total={rt:5.2f}s  ({feats.duration_s/rt:5.1f}x realtime)"
            )
            print(f"      noise_floor={feats.noise_floor_db:6.1f}dBFS  gate={scorer.floor_gate_db:6.1f}dBFS")
            print(f"      BEST   {explain(scorer, best)}")
            for a in alts[:3]:
                print(f"      alt    {explain(scorer, a)}")
    if n:
        print("=" * 108)
        mean = total_t / n
        print(f"{n} files, mean {mean:.2f}s/file -> 5558 files on 15 workers "
              f"= {5558*mean/15/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
