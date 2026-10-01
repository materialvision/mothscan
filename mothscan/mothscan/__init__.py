"""mothscan - find the acoustically most interesting moment in every AudioMoth file.

Quick start
-----------
    from mothscan.config import AnalysisConfig
    from mothscan.features import extract_features
    from mothscan.scoring import EventScorer

    feats  = extract_features("20260520_150000.WAV", AnalysisConfig())
    scorer = EventScorer(feats, AnalysisConfig())
    best, alts = scorer.find_best()
    print(best.start_s, best.score)
"""

from .config import AnalysisConfig, PreviewConfig, RunConfig, SITE_DIRS, SITE_ROOT

__all__ = [
    "AnalysisConfig",
    "PreviewConfig",
    "RunConfig",
    "SITE_DIRS",
    "SITE_ROOT",
]
__version__ = "0.1.0"
