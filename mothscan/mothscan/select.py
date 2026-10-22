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
    contam_cfg: ContaminationConfig | None = None,
    classes: dict[int, Contamination] | None = None,
    # Accepted for call-site compatibility; substitution is no longer performed.
    sub_cfg: SubstitutionConfig | None = None,
) -> list[SelectedPoint]:
    """Pick one point per hour. Each hour plays its own recording.

    ``site_slots`` maps grid index -> record for the site, ``indicators`` maps
    grid index -> anything :func:`classify` accepts (a FileFeatures, normally).
    Pass ``classes`` to supply verdicts directly and skip classification.
    """
    contam_cfg = contam_cfg or ContaminationConfig()

    if classes is None:
        classes = {g: classify(ind, contam_cfg) for g, ind in indicators.items()}
    scores: dict[int, float] = {g: c.score for g, c in classes.items()}

    # One point per hour, and it always plays *its own* recording.
    #
    # Borrowing a neighbour's hour existed to avoid deliberately listening to
    # wind rumble or a railed microphone. It caused a worse problem than it
    # solved: a neighbour's recording played in two slots (the borrowed hour and
    # the neighbour's own), which is the repeated clip that was reported, and it
    # was hard to see because it only happened for windy hours whose donor was
    # itself in the stream.
    #
    # Contamination is still detected, recorded and reported - it shows as amber
    # tags in the viewer and as the `noise` column in the exports - but it no
    # longer changes which audio plays. A windy hour sounds windy, honestly.
    # With this, "no recording is heard twice" holds by construction: slot i
    # plays recording i, and the recordings are distinct.
    out: list[SelectedPoint] = []
    for grid in sorted(site_slots):
        own = site_slots[grid]
        mine = classes.get(grid, Contamination())
        out.append(
            SelectedPoint(
                grid_index=grid,
                site=site,
                record=own,
                own_record=own,
                contamination=mine,
                substituted=False,
                origin_grid=None,
                origin_name=None,
                substitute_score=0.0,
                reason="",
            )
        )
    out.sort(key=lambda p: p.grid_index)
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
