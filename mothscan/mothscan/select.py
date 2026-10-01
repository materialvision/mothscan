"""Choose the final point per hour, including nearby-hour substitution.

Why this is a separate pass
---------------------------
The scorer answers *"where inside this recording is the most interesting 0.5 s?"*
It cannot answer *"is this recording worth using at all?"*, because that is a
question about the **whole** file - a wind-dominated hour can still contain one
locally eventful moment, and picking it would be misleading.

So selection runs in two stages:

1.  score every hour for its best window (the hybrid event score);
2.  classify each recording for wind / sustained clipping, and where a recording
    is unusable, look for the nearest acceptable hour and take *that* hour's
    excerpt instead, recording where it really came from.

The substitution is explicit in the output: the chosen point keeps the identity
of the hour it *represents* (for the timeline and for the export's grid column)
but carries ``origin_name`` / ``origin_grid`` naming the recording the audio
actually came from.  Nothing is silently swapped.

Substitution is bounded and conservative
----------------------------------------
* only a limited window of hours is searched (``max_distance_h``);
* a substitute must be meaningfully cleaner, not merely different
  (``min_score_gain``);
* a substitute may not itself be contaminated beyond
  ``max_substitute_score``;
* if nothing suitable is found the hour is still emitted, flagged
  ``substituted=False`` with ``noise_score`` set, so it can be filtered out
  downstream rather than quietly disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .noise import (
    Contamination,
    ContaminationConfig,
    SubstitutionConfig,
    classify,
    find_substitute,
)


@dataclass
class SelectedPoint:
    """One hour's chosen excerpt, possibly borrowed from a nearby hour."""

    grid_index: int
    site: str
    #: The record whose audio is used (a neighbour's, when substituted).
    record: object
    #: The record for the hour itself, whether or not it was used.
    own_record: object | None
    contamination: Contamination
    substituted: bool = False
    origin_grid: int | None = None
    origin_name: str | None = None
    substitute_score: float = 0.0
    reason: str = ""

    @property
    def own_score(self) -> float:
        return self.contamination.score


def contamination_from_record(rec) -> Contamination:
    """Rebuild a verdict from the flags a record already carries.

    Re-classifying from the index is not possible: the stored record keeps the
    *conclusion* (wind / clipped / sustained) plus the score, not the raw
    indicators.  Using the stored conclusion also guarantees the selection pass
    and the index agree - there is exactly one place that decides.
    """
    return Contamination(
        score=float(getattr(rec, "noise_score", 0.0) or 0.0),
        wind=bool(getattr(rec, "noise_wind", False)),
        clipped=bool(getattr(rec, "noise_clipped", False)),
        sustained=bool(getattr(rec, "noise_sustained", False)),
        clip_frac=float(getattr(rec, "clip_frac", 0.0) or 0.0),
    )


def select_points(
    site: str,
    site_slots: dict[int, object],
    indicators: dict[int, object],
    sub_cfg: SubstitutionConfig | None = None,
    contam_cfg: ContaminationConfig | None = None,
    classes: dict[int, Contamination] | None = None,
) -> list[SelectedPoint]:
    """Pick one point per hour, substituting from a neighbour when unusable.

    ``site_slots`` maps grid index -> record for the site, ``indicators`` maps
    grid index -> anything :func:`classify` accepts (a FileFeatures, normally).
    Pass ``classes`` to supply verdicts directly and skip classification.
    """
    sub_cfg = sub_cfg or SubstitutionConfig()
    contam_cfg = contam_cfg or ContaminationConfig()

    if classes is None:
        classes = {g: classify(ind, contam_cfg) for g, ind in indicators.items()}
    scores: dict[int, float] = {g: c.score for g, c in classes.items()}

    used_substitutes: set[int] = set()
    out: list[SelectedPoint] = []
    for grid in sorted(site_slots):
        own = site_slots[grid]
        mine = classes.get(grid, Contamination())
        rec = own
        substituted = False
        origin_grid = None
        origin_name = None
        sub_score = 0.0
        reason = ""

        if mine.unusable and sub_cfg.mode != "silence":
            found = find_substitute(site_slots, grid, scores, sub_cfg, contam_cfg,
                                    used=used_substitutes)
            if found is not None:
                # Record the borrow even when it was a fallback, so the next
                # wind-dominated hour still prefers a different neighbour.
                used_substitutes.add(found[0])
                g2, other = found
                rec = other
                substituted = True
                origin_grid = g2
                origin_name = getattr(other, "source_name", None)
                sub_score = scores.get(g2, 0.0)
                why = []
                if mine.wind:
                    why.append("wind-dominated")
                if mine.sustained:
                    why.append("sustained clipping")
                reason = (
                    f"{' and '.join(why) or 'contaminated'}; took hour "
                    f"{g2 - grid:+d} instead"
                )

        out.append(
            SelectedPoint(
                grid_index=grid,
                site=site,
                record=rec,
                own_record=own,
                contamination=mine,
                substituted=substituted,
                origin_grid=origin_grid,
                origin_name=origin_name,
                substitute_score=sub_score,
                reason=reason,
            )
        )
    return out


def summarise(points: list[SelectedPoint]) -> dict:
    """Counts for a run report."""
    n = len(points)
    sub = sum(1 for p in points if p.substituted)
    wind = sum(1 for p in points if p.contamination.wind)
    clip = sum(1 for p in points if p.contamination.clipped)
    sust = sum(1 for p in points if p.contamination.sustained)
    return {
        "hours": n,
        "substituted": sub,
        "wind": wind,
        "clipped": clip,
        "sustained_clip": sust,
        "substituted_pct": round(100.0 * sub / n, 2) if n else 0.0,
    }
