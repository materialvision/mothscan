# mothscan — finding the interesting 0.5 second in every AudioMoth hour

A framework for scanning an AudioMoth deployment, picking the acoustically most
eventful moment in each 2-minute recording, stitching those moments into one
continuous pulse-train preview you can listen through, viewing the result as a
waveform + spectrogram in the browser, and exporting the cue points to
Max/MSP `coll`.

---

## 1. What is in the dataset

| | |
|---|---|
| Sites | `moth oculus`, `moth-juvet`, `moth-kvarvet` under `/Volumes/Solardisk/Harpefoss` |
| Files | 5,558 WAV (2,376 / 2,377 / 805) |
| Format | 48 kHz, 16-bit PCM, **mono**, 120 s each — AudioMoth "Basic" firmware |
| Cadence | one 120 s file per hour, 24 h/day (`sleep 3480`, period 00:00–24:00 UTC) |
| Span | 2026-05-20 → 2026-08-27 (~99 days per site) |
| Size | 64 GB |
| Discarded | 5,558 `.reapeaks` files (REAPER waveform caches next to every WAV that was dragged into REAPER). `config.is_audio_file()` rejects them by suffix, so they never enter the pipeline. |
| Oddities | 6 partial files (deployment/retrieval truncation: 6.1 s, 5.1 s, 31.1 s, and three ~120 s-minus-a-few-ms). They are analysed and flagged, not dropped. |

---

## 2. The analysis framework

The question "where is a good audio event?" is answered by a **hybrid,
self-normalising window score**. Each part is deliberately chosen to cover the
failure mode of the others.

### 2.1 Representation

* STFT: `n_fft = 1024`, hop `256` → one analysis frame every **5.33 ms**.
  That is fine enough to resolve a raindrop's attack and a wingbeat click, and
  cheap enough to process 64 GB in under two minutes.
* 48 **mel** bands, 40 Hz – 20 kHz. Mel spacing matches how the content is
  distributed (most biological energy is low; insect song is high).

### 2.2 The four (five) terms

| term | window statistic | what it finds |
|---|---|---|
| `novelty` | mean **spectral flux** over the window | birdsong phrases, insect stridulation, rain onsets — anything that *changes* |
| `transient` | peak **envelope attack** in the window | raindrops, clicks, wing claps, sudden calls — anything that *strikes* |
| `contrast` | log ratio of the window's loudest envelope to its own 400 ms background | does the event **stand out** from its surroundings |
| `flatness` | mean spectral flatness (geometric/arithmetic mean) | broadband events (rain, rustle) that a flux detector under-serves |
| `tonality` | 1 − flatness, weight 0 by default | optional bias toward tonal content (birds, insects) |

**Novelty** is not raw flux. Per mel band we subtract a percentile of that
band's positive-flux distribution before rectifying. This is what makes the
detector work in both a silent forest night *and* a downpour: stationary
broadband noise contributes ≈0 novelty, while any band that changes relative to
its own floor produces a positive excursion. Quantile normalisation of the
spectrogram is applied exactly where it does the most work.

**Contrast** is the term that rescues a quiet bird in a quiet hour while
refusing a loud-but-constant rain drone — the single most important guard
against the "loudest window" trap.

### 2.3 Normalisation (the detail that matters most)

Every term is divided by a high percentile of **its own windowed distribution
across the same file**, then clipped at 3.0.

Normalising on *window* statistics rather than raw frame statistics is
essential: a 120 s recording is mostly empty frames, so a per-frame percentile
sits inside the noise and every window ratio explodes. On the windowed scale a
term of 1.0 means *"about as eventful as the busiest part of this hour"*, which
is exactly the comparable scale needed when comparing a 3 a.m. winter hour
against a noon chorus.

### 2.4 Event-centred search

Scoring and placement happen in one search rather than being bolted together
afterwards, so the reported `cut_s` is directly usable as a cut point and an
event near a file boundary shifts the window instead of being lost.

### 2.5 Where the event sits in the excerpt

The placement rule is an **absolute pre-roll**, not a fraction of the analysed
window. That distinction is what makes the excerpt length a free render-time
choice.

```
event offset into the excerpt = max(event_lead_s, 10 % of the excerpt)
```

| excerpt length | event sits at | headroom after the event |
|---|---|---|
| 2.0 s | 200 ms | 1.80 s |
| 0.5 s | 50 ms | 0.45 s |
| 0.2 s | 20 ms | 0.18 s |
| 0.1 s | 10.7 ms | 0.09 s |
| 0.05 s | 10.7 ms | 0.04 s |

Why not a fraction of the window: `0.35 × 0.5 s = 176 ms` is a *fixed number of
milliseconds*, so shortening the cut to 0.2 s left the event 88 % of the way in
with 24 ms to spare, and below ~0.18 s **the event fell outside the excerpt
entirely** — 0 % of hours contained their event at a 0.15 s cut. Measured on the
old arrangement:

```
 cut_s   event inside excerpt
  0.50         100.0%
  0.20         100.0%   (24 ms of headroom)
  0.15           0.0%   <- the event is gone
  0.10           0.0%
  0.05           0.0%
```

Two mechanisms now guarantee containment:

* **At render time**, `preview.excerpt_start()` re-places each excerpt from the
  record's own `event_s`, so whatever length you ask for, the attack lands just
  inside the start. Nothing needs re-analysing — change the cut and re-render.
* **A static pre-pass** (`check_events_contained`) verifies the invariant across
  a whole site before rendering and reports violations to stderr, so a future
  change that breaks it is loud rather than silent.

The pre-roll is deliberately tiny (10.7 ms ≈ two analysis frames, floored at 10 %
of short excerpts) for two reasons: it survives very short cuts, and it keeps the
attack clear of the excerpt's own 4 ms fade-in so the onset still reads as an
onset rather than being softened.

The index stores both coordinate pairs, because they are genuinely different
things and conflating them caused a bug:

| | this hour's own recording | the audio actually used |
|---|---|---|
| file | `source_name` | `audio_name` (= `origin_name` when substituted) |
| cut | `cut_s` | `audio_cut_s` |
| event | `event_s` | `audio_event_s` |

A substituted hour reports *its own* event in the viewer, but is *placed* using
the borrowed recording's event — mixing the two put the excerpt around an event
that was not in the file being read.

### 2.6 Loudness levelling (the limiter)

A train passing the recorder sits ~36 dB above a distant warbler, so a plain
concatenation is unpleasant to scan: the loud hours force you to ride the volume
and the quiet ones vanish. The preview is therefore levelled with a **look-ahead
peak limiter**, which only ever *reduces* peaks and never lifts quiet material.

That asymmetry is deliberate. Normalising each excerpt to a common peak would
make the birds as loud as the train — inverting the relationship and inventing
level that was never recorded. Reducing only the peaks keeps every event's level
relative to every other event, which is the property that makes the scan
readable as a survey.

Measured on `20260818_120000.WAV` (the train):

```
                              raw        limited
loudest 0.5 s (the train)    -0.00 dBFS   -12.00 dBFS
quietest 0.5 s              -35.60 dBFS   -35.60 dBFS   (untouched)
dynamic range                 35.6 dB       23.6 dB
```

Across a sample of hours the reduction is exactly 0.00 dB for anything under the
knee, so quiet hours are bit-identical to the raw material.

| setting | default | effect |
|---|---|---|
| `limiter` | on | master switch; `--no-limiter` renders raw |
| `limiter_ceiling_dbfs` | −12.0 | peak ceiling. Raise toward 0 dBFS to hear the raw material; lower to −20 for a very flat scan |
| `limiter_knee_db` | 6.0 | soft knee, so quiet material is untouched |
| `limiter_lookahead_s` | 0.005 | **centred** look-ahead, so the gain is already down when a transient arrives |
| `limiter_release_s` | 0.12 | recovery time; longer = smoother but more ducking |

The look-ahead is *centred* rather than causal, which makes the limiter
**zero-latency**: a causal look-ahead would delay the audio and shift every event
inside its excerpt, undoing the placement guarantee in §2.5. There is a test for
this — the limiter must cap the rendered peak without moving the signal.

### 2.7 Overlapping the excerpts

A **negative gap** makes consecutive excerpts overlap: the next one starts while
the previous is still sounding. At a 0.5 s cut, `gap = -0.25` means each excerpt
begins 0.25 s after the last and 0.25 s of every pair plays together.

The number of excerpts sounding at once is simply **`cut ÷ slot pitch`**, where
`slot pitch = cut + gap`:

| gap | slot pitch | clips overlapping |
|---|---|---|
| 0.0 | 0.500 s | 1 |
| −0.1 | 0.400 s | 1.25 |
| −0.25 | 0.250 s | 2 |
| −0.3 | 0.200 s | 2.5 |
| **−0.375** | 0.125 s | **4** |
| **−0.4** | 0.100 s | **5** |
| −0.45 | 0.050 s | 10 |
| −0.49 | 0.010 s | 50 |

Note the arithmetic: at a 0.5 s cut, `gap = −0.4` gives **five** excerpts
overlapping, not four — four needs `gap = −0.375`. Measured on a real render by
counting, at every instant, how many excerpts are sounding:

```
gap=-0.375  pitch=0.125s  cut/pitch=4.00  observed: max=4 median=4
gap=-0.400  pitch=0.100s  cut/pitch=5.00  observed: max=5 median=5
```

**The limit** is not a fixed cap on the count: it is that the slot pitch must
stay **positive**, i.e. `gap > −cut`. The count is `cut / pitch`, so as the pitch
approaches zero the overlap grows without bound — at a 0.5 s cut the tightest
legal setting is `gap ≈ −0.5`, where the count diverges. A pitch of zero or less
is refused (HTTP 400) because the grid exists to keep the sites aligned and a
non-positive pitch destroys that. The viewer's gap slider is bounded to
`−cut + 0.01` for the same reason, so it cannot even request an impossible value.

`crossfade` is the fade on each excerpt edge, and when excerpts overlap it *is*
the crossfade — raise it for a softer blend. It is capped at `cut / 2`, because
if the fade-in and fade-out could meet they would squash the middle of the
excerpt to nothing and the event, which sits a short pre-roll in, would be the
first thing lost.

Three things needed care:

* **Clips must sum, not overwrite.** The buffer is accumulated with `+=`. Writing
  each excerpt over its slot would chop the previous tail off rather than blend
  the two, which is the whole point of the mode.
* **The buffer must allow the overhang.** An excerpt now runs past its own slot,
  so `total_samples` is `(n-1) * pitch + cut` rather than `n * pitch`; otherwise
  the last excerpts would be truncated.
* **The limiter had to move after the mix.** Two summed excerpts can exceed
  either alone, and a per-excerpt limiter cannot see that. It now runs once over
  the mixed buffer, which still follows each sound because the reduction comes
  from the signal's own envelope.

**Site alignment is untouched.** Slots do not move: `stream_start` is still
`index * pitch`, so slot *i* is at the same offset in all three streams whatever
the gap. Only the advance *between* slots changes. Verified at `gap` = −0.10,
−0.25 and −0.40: 2,379 slots checked, 0 mismatches, identical durations.

A slot pitch of zero or less is refused rather than rendered, because it would
destroy the alignment the grid exists to provide.

**Levels.** Overlapping necessarily sums, so `overlap_scale` controls what that
does:

| value | effect |
|---|---|
| `"auto"` (default) | scale by `1/sqrt(overlap)`, so piling clips up keeps the overall loudness roughly constant. No effect when they do not overlap, so a normal render is unchanged. |
| `"unity"` | let overlaps add at full level — a dense passage simply gets louder. The limiter still guarantees no clipping. |
| a number | fixed gain. |

**It also shortens the stream**, which is a useful side effect: 99 days of hourly
clips at a 0.5 s cut is 21.8 min per site sequentially but 9.9 min at
`gap = -0.25`.

### 2.8 Speed: what it does, and where to do it

There are **two** speed controls, and they do different things.

**`play at` above the timeline (live, recommended).** Sets the audio element's
`playbackRate` and asks the browser to preserve pitch. Nothing is re-rendered, it
is instantly adjustable, and it uses the browser's own production-grade time
stretch — measurably better than anything this tool renders offline. For
skimming 99 days this is the one to use.

**`Bake speed into the render` (offline).** Changes the *file*: each excerpt
occupies `cut / speed` seconds instead of `cut`. Useful when the stream leaves
the browser (a Max patch, a DAW session, a shared file). Two modes:

| mode | duration | pitch | transparency |
|---|---|---|---|
| `varispeed` (default) | × 1/speed | follows speed | exact — it is pure resampling |
| `pitch` | × 1/speed | preserved | good, not perfect (WSOLA overlap-add) |

**It compresses the whole stream**, slot pitch included: at 2× the pitch becomes
`(cut + gap) / 2`, so a 0.5 s excerpt fills its 0.275 s slot and the pulse train
stays gapless. The stream is genuinely half as long.

An earlier version scaled only the audio *inside* each slot and left the spacing
alone. Every slot then became 0.25 s of audio followed by 0.30 s of silence —
audibly a click train rather than a sped-up scan, which is exactly the bug that
was reported.

| speed | grid pitch (wall clock) | stream pitch | oculus stream |
|---|---|---|---|
| 1× | 0.550 s | 0.550 s | 21.8 min |
| 2× | 0.550 s | 0.275 s | 10.9 min |
| 4× | 0.550 s | 0.138 s | 5.5 min |

Both pitches are reported, because they are different quantities:
`grid_slot_seconds` is wall-clock time (one grid slot = one hour), and each
site's `slot_seconds` is the spacing actually heard. From these,
`stream_ms = (grid − grid_origin) × slot_seconds × 1000`.

**Alignment survives.** The grid index of a slot is unchanged, and every site is
compressed by the same factor, so slot *i* still sits at the same stream offset
in all three streams. Verified at 2×: 600 slots, 0 mismatches, identical
durations.

**Exports must be told the speed**, since `stream_ms` is what Max seeks with:
`mothscan export --speed 2` (the viewer sends it automatically). Omitted, the
`coll` would point into the middle of a later excerpt. The `grid` column is
unaffected either way, so the cross-site alignment key stays valid.

#### What was wrong before

The original `time_compress` was plain polyphase resampling, described in its own
docstring as "pitch-preserving". It was the opposite, and wrong twice over:

```python
up, down = int(round(speed * 100)), 100
resample_poly(x, up // g, down // g)     # 2x -> half the samples...
```

Halving the sample count and then declaring the buffer at the *original* rate
gives double the duration at half the pitch. Measured on a 1.000 s 2 kHz tone:

| speed | duration | pitch |
|---|---|---|
| 1.0 | 1.000 s | 2000 Hz |
| **2.0** | **2.000 s** (wanted 0.500 s) | **1000 Hz** (wanted 2000 Hz) |
| 0.5 | 0.500 s (wanted 2.000 s) | 4000 Hz |

So "2×" went half as fast an octave down, and `varispeed` had the ratio inverted
for the same reason. That is exactly the "like a low sample rate" impression —
because it *was* one. Both directions are now correct and covered by unit tests
(`tests/test_preview.py`), and the `speed` control is actually sent by the UI,
which it never was.

### 2.9 Gates and penalties

* **Activity gate** — the window must exceed the file's own noise floor
  (5th percentile of frame RMS) by 15 dB, and exceed −62 dBFS absolutely.
  Files whose best window still fails are kept but flagged
  (`passes_gate = false`), so silent hours appear as silent pulses in the
  preview rather than silently vanishing from the timeline.
* **Edge penalty** — handling noise, wind bursts and the recorder's power-up
  transient cluster at the first and last second of an unattended deployment
  file, so windows near an edge are multiplied by 0.5 → 1.0 as they move inward.
* **Coarse step 25 ms, refine 5 ms** around the coarse winner. Integral images
  make the coarse cost independent of window length.

### 2.10 Verification, not faith

`scripts/diagnose.py` renders, for any file, a three-panel figure: the full
mel spectrogram, the score curve over the whole file, and the waveform — with
the chosen cut window, the detected event and the top alternatives marked. That
is how the model was tuned, and it is how you should re-tune it:

```bash
MPLCONFIGDIR=data/.mplcache .venv/bin/python scripts/diagnose.py --n 9
# -> data/diag/*_diag.png  plus  *_compare.wav  (picked | loudest | random, back to back)
```

Seen in those diagnostics: repeated insect call trains, rain events, sharp
broadband transients, and bird chirps — all correctly located.

---

## 2b. Wind noise and microphone clipping

Wind violates the premise of an "interesting event" detector: a gust is
spectrally novel, has a strong envelope, and stands far above the local
background, so the hybrid score *loves* it. Meanwhile clipping is a sampling
defect, not content. Both are therefore detected explicitly and handled
separately — and because both are **physical** signatures, they can be measured
rather than guessed at.

### The indicators

| indicator | wind | clipping | why it discriminates |
|---|---|---|---|
| `low100` | **> 0.30** | — | fraction of energy below 100 Hz. This deployment left the AudioMoth's optional 48 Hz DC-blocking filter **disabled**, so the band is intact. Strongly bimodal across the deployment: p50 = 0.04, p75 = 0.10, then a jump to p90 = 0.41 — 0.30 sits in that empty middle. |
| `tilt_db_oct` | **< −4.0** | — | long-term spectral slope. Wind collapses with frequency; rain and an insect chorus are comparatively flat. |
| `peak_db` | — | **≥ −0.5 dBFS** | at the 16-bit rail. 463 files (8.3 %) sit exactly at full scale. Being *at* the rail is clipping, full stop. |
| `clip_run` | — | **≥ 12** | longest run of identical PCM samples. Quantisation noise gives runs of 2–3; a rail gives long flat tops. Validated: every file at the rail had a run ≥ 12 and **no** file below −1 dBFS did, so this test has no false positives here. |
| `sustain` | 0.30 | — | fraction of frames within 6 dB of the loudest. Wind holds one level for minutes; birdsong is eventful. Strengthens the verdict, does not gate it. |

Classification is **conjunctive** — the low band must dominate *and* the whole
spectrum must tilt — so a genuine low-frequency animal (bittern, frog chorus) is
not dismissed as weather.

The thresholds are not guesses. `scripts/calibrate_noise.py` prints the
deployment's own indicator distributions and `scripts/validate_noise.py` renders
labelled galleries of the wind / clipped / clean populations on a common dB
scale, so the decision boundary can be checked by eye.

### Two different responses

**1. Don't cut there.** A window whose samples sit on the converter rail is
attenuated to 5 % of its score: a clipped excerpt is a defect, so it is rejected
rather than softly penalised. This is applied *per window* from a per-frame rail
curve, because clipping is localised — one pinned wingbeat should not condemn an
otherwise excellent hour.

**2. If the hour is unusable, play a nearby hour instead.** Wind is not
localised, so a wind-dominated hour cannot be rescued by moving the window. For
those, the picker takes the **nearest acceptable neighbour's** excerpt and
records where it came from:

```
hour 494296 (2026-05-22 16:00Z)  own file 20260522_160000.WAV, own cut 54.869 s
   noise score 0.46, wind-dominated
   -> plays 20260522_170000.WAV at its own cut 46.261 s   (a different hour, +1)
   verified: rendered low-band share 8.0 % (clean) instead of 48.6 % (windy)
```

The substitution is **explicit, never silent**:

* the point keeps its own hour for the timeline and the export's `grid` column;
* `cut_s` / `path` describe the audio actually used (the neighbour's);
* `origin_grid` / `origin_name` / `substitution_reason` name where it came from;
* it is bounded (default ±6 h), requires the substitute to be meaningfully
  cleaner, and a neighbour is not reused by consecutive hours — a storm lasts
  hours, and without that rule a whole windy night would replay one excerpt.

### Result across this deployment

```
hours taking a neighbour's excerpt: 368 of 5558  (6.62 %)
  moth oculus    213 / 2376      wind 203, rail-touch 254, sustained clipping 13
  moth-juvet       1 / 2377      wind   1, rail-touch 124
  moth-kvarvet   154 /  805      wind 161, rail-touch 161
```

The exposure is exactly where you would expect it: `moth-juvet` is essentially
wind-free (one hour), while the two exposed sites lose ~7–19 % of their hours.

Of 365 wind-flagged hours, **355 took a neighbour**; the remaining 10 sit inside
multi-day storms where *every* hour within range is also windy. There is no clean
alternative to borrow, so they keep a high `noise` score and `substituted = 0` —
flagged, not quietly filled with equally bad audio. Filter them in Max with
`column 11 <= 400 and column 12 == 0`.

Mean contamination of the hours actually presented falls to **0.064 / 0.035 /
0.146** (oculus / juvet / kvarvet).

### In the viewer

* Substituted hours carry an amber tag along the top of the lane, and the point
  panel reports *"took hour 1h earlier — plays 20260522_170000.WAV"* plus the
  contamination score.
* Hours with no recording stay red-banded, as before.

---

## 3. Results of the full run

```
5,558 files analysed, 0 errors  (~23 files/s on 15 workers, ~240 s;
                                 the contamination indicators cost ~2x the feature pass)
  moth oculus     2376 points,  1049 passed the activity gate ( 44.1%), mean score 3.71
  moth-juvet      2377 points,  1600 passed the activity gate ( 67.3%), mean score 4.21
  moth-kvarvet     805 points,   618 passed the activity gate ( 76.8%), mean score 3.36

  wind-dominated: 365    touched the converter rail: 539    sustained clipping: 13
  nearby-hour substitution: 368 hours (6.62%)
```

Artefacts:

```
data/mothscan.sqlite3             queryable index (points + runs)
data/points.jsonl                 5,558 lines, one selected point each, chronological
data/cache/<site>/*.npz           cached feature curves -> re-tune without the drive
data/noise_calibration/           indicator distributions + flagged population counts
data/noise_validation/            labelled spectrogram galleries (wind/clipped/clean)

data/previews/scan.index.json     the shared grid + per-site summaries + gaps
data/previews/scan_<site>.wav     ONE STREAM PER SITE, all 21.81 min, sample-aligned
data/previews/scan_<site>.spec    browser spectrogram (uint8)
data/previews/scan_<site>.env     browser waveform envelope (float16)
data/previews/scan_<site>.slots.jsonl   one JSON object per hour slot

data/exports/points_<site>.coll   Max/MSP cue table per site (12 columns)
data/exports/points.coll          all sites combined, with a site column
data/exports/cuepoints_<site>.coll  stream_ms + duration_ms
data/exports/points_<site>.csv    per site, for Numbers/Excel
data/exports/points.csv           all sites
data/exports/points_<site>.dict.json  Max [dict], per site
data/exports/max_notes.txt        coll column layout + working Max recipes
```

---

## 3b. Three sites, three streams, one clock

The three locations are kept **as separate files and streams**, but they are
placed on **one shared hourly grid** so they stay in sync by date and hour.

### The grid

Every selected point gets an absolute slot index:

```
slot = (timestamp truncated to the hour - epoch) / 3600 s
```

The grid pitch in the rendered stream is `cut + gap` (default 0.60 s with the
1.0 s example below; 0.55 s at the defaults). Because the pitch is fixed and all
sites share one **grid origin**, site A's stream offset for hour *H* equals site
B's exactly. Verified: the three 21.81-minute WAVs are all 20,935,200 samples,
and the maximum deviation of any slot's offset from `index × slot_seconds` over
1,200 sampled slots is 2.8 × 10⁻¹⁴ s.

```
grid 494247  →  15:00 UTC on 2026-05-20, at stream 1.650 s in ALL three streams
   moth oculus  stream 1.650 s  ← 20260520_150000.WAV
   moth-juvet   stream 1.650 s  ← 20260520_150000.WAV
   moth-kvarvet stream 1.650 s  ← 20260520_150000.WAV
```

### Missing hours — your choice, per render

| policy | behaviour |
|---|---|
| `silence` **(default)** | an hour with no recording becomes a silent slot, so the stream's offsets stay identical to the grid and the sites remain aligned for the whole deployment |
| `skip` | the slot is dropped, giving a shorter gapless stream; the grid index is still recorded per surviving slot, so an export can re-place them |

Set it in the CLI (`preview --missing skip`), in the viewer (*Hours with no
recording*), or in the API (`"missing_policy": "skip"`).

In this dataset there are **no unexpected gaps** — each deployment is complete
over its own window. The one real alignment case is that `moth-kvarvet` stops
after 805 hours while the other two run for 2,377:

```
moth oculus    21.81 min   2376 audio,    3 silent   (before its first file)
moth-juvet     21.81 min   2377 audio,    2 silent   (before its first file)
moth-kvarvet   21.81 min    805 audio, 1574 silent   (after its deployment ended)
```

So `moth-kvarvet`'s lane visibly ends two thirds of the way across the timeline.
That silent tail is what buys exact three-way alignment; `--no-align` gives each
site its own origin instead (kvarvet becomes 7.38 min, but the offsets are then
comparable only per site).

### What the viewer does with this

* Three lanes stacked on one shared time axis, one per site, each with its own
  colour. A lane with no data for an hour is drawn as an explicit
  **red-banded "no recording"** region rather than left as ambiguous silence.
* **Switching site keeps the playhead on the same wall-clock hour** — press
  *play this site* on any lane and the same hour starts playing there. That is
  the A/B comparison across locations.
* Hovering any lane reports the **absolute hour** (`2026-08-09 01:00:00 UTC`),
  the grid index, the source filename, and the cut position inside it.
* Export scope is a selector: *active site only* or *all sites*.

---

## 4. Using it

### 4.1 Setup and running it

`mothscan.sh` is the entry point. It resolves its own location, so it works from
anywhere, and it finds a usable Python for you.

**One-time setup** — creates the venv, installs the dependencies, then *imports
them to prove it worked* rather than assuming:

```bash
cd mothscan
./mothscan.sh setup
```

**Everyday use:**

```bash
./mothscan.sh start      # start the viewer in the background
./mothscan.sh status     # is it running?  (exit 0 = yes, 1 = no — scriptable)
./mothscan.sh logs       # tail the viewer log
./mothscan.sh stop       # stop it
./mothscan.sh restart    # stop then start

./mothscan.sh open       # start if needed, print the URL, open a browser

./mothscan.sh serve      # run in the FOREGROUND instead (ctrl-c to stop)
```

`start` is idempotent: if something is already answering on the port it says so
and does nothing, so you cannot accidentally run two servers on one port. It also
refuses to report success until the API actually responds, and if the process
dies immediately it points you at `data/viewer.log` instead of leaving you
guessing.

**Analysing and rendering:**

```bash
./mothscan.sh scan --limit 200    # try a small sample first
./mothscan.sh stats               # coverage, gate rates, top points
./mothscan.sh preview             # render the per-site preview streams
./mothscan.sh export              # write the Max/MSP coll files
./mothscan.sh build               # all three, end to end
./mothscan.sh test                # unit tests (no data or server needed)
./mothscan.sh verify              # 34 headless behavioural checks on the viewer
```

**Configuration** — all optional, all environment variables:

| variable | default | meaning |
|---|---|---|
| `MOTHSCAN_SITE_ROOT` | `/Volumes/Solardisk/Harpefoss` | where the recordings live |
| `MOTHSCAN_SITES` | the three Harpefoss names | comma-separated site directories |
| `MOTHSCAN_DATA_DIR` | `<repo>/data` | caches, index, renders (can be many GB) |
| `MOTHSCAN_VENV` | first of `.venv`, `../.venv` | which Python environment to use |
| `MOTHSCAN_PORT` | `8770` | viewer port |

```bash
export MOTHSCAN_SITE_ROOT=/path/to/recordings
export MOTHSCAN_SITES="site-a,site-b,site-c"
./mothscan.sh start
```

If no usable Python is found, the wrapper says exactly which paths it tried and
how to fix it, rather than failing with an import traceback.

### 4.2 The Python CLI directly

The wrapper is a thin front-end; everything is also reachable as
`python -m mothscan.cli <command>`, which is what to use when scripting or when
you want the full flag list:

```bash
P=./.venv/bin/python        # or ../.venv/bin/python on the original checkout

$P -m mothscan.cli scan                      # analyse everything -> index (resumable)
$P -m mothscan.cli scan --limit 500           # quick trial
$P -m mothscan.cli scan --reset               # start over
$P -m mothscan.cli stats                      # coverage, gate rates, top points
$P -m mothscan.cli inspect 20260709_020000.WAV  # why THIS moment? full score breakdown
$P -m mothscan.cli pick 20260709_020000.WAV     # write just that excerpt as a wav
$P -m mothscan.cli preview --cut 0.5 --gap 0.05 --rate 16000
$P -m mothscan.cli export --cut 0.5 --gap 0.05
$P -m mothscan.cli serve                      # browser viewer on :8770

# wind / clipping handling
$P -m mothscan.cli scan --refresh              # recompute cached features (after a detector change)
$P -m mothscan.cli scan --substitute both      # default: nearest acceptable hour
$P -m mothscan.cli scan --substitute silence   # flag bad hours instead of replacing them
$P -m mothscan.cli scan --substitute-window 12 # search further for a clean hour
$P -m mothscan.cli scan --no-substitute        # leave every hour as it is

# one site only
$P -m mothscan.cli preview --sites "moth oculus" --name oculus_only
$P -m mothscan.cli export  --sites "moth-kvarvet" --per-site-only

# short excerpts: the event is re-placed just inside the start, never lost
$P -m mothscan.cli preview --cut 0.2 --gap 0.05

# overlapping: a dense, continuous texture instead of a pulse train
$P -m mothscan.cli preview --cut 0.5 --gap -0.25 --fade 0.06
$P -m mothscan.cli preview --cut 1.0 --gap -0.9  --fade 0.25   # almost fully continuous
$P -m mothscan.cli preview --cut 0.5 --gap -0.25 --overlap-scale unity

# loudness levelling
$P -m mothscan.cli preview --ceiling -18      # flatter, quieter scan
$P -m mothscan.cli preview --ceiling -3       # nearer the raw material
$P -m mothscan.cli preview --no-limiter       # completely raw

# missing hours: pad with silence (aligned) or skip (shorter, gapless)
$P -m mothscan.cli preview --missing silence   # default
$P -m mothscan.cli preview --missing skip
$P -m mothscan.cli preview --no-align          # each site starts at its own first hour
```

Re-tuning the model costs seconds, because every file's feature curves are
cached:

```bash
# favour broadband/transient events
$P -m mothscan.cli retune --weights transient=2 novelty=1 contrast=0.6 flatness=0.5
# favour tonal content (birds, insect song)
$P -m mothscan.cli retune --weights novelty=1.2 tonality=1.0 transient=0.4
$P -m mothscan.cli preview && $P -m mothscan.cli export   # hear/export the new picks
```

`scan --weights ...` does the same thing in one pass on a fresh run.

### 4.3 Browser viewer

```bash
./mothscan.sh open                  # start if needed, print URL, open a browser
./mothscan.sh start                 # or just start it: http://127.0.0.1:8770
./mothscan.sh logs                  # watch what it is doing
./mothscan.sh stop                  # when done
```

* **Three lanes, one per site**, stacked on a single shared time axis, each in
  its own colour, with a live playhead across all of them.
* Each lane shows its **spectrogram** of the whole deployment plus its waveform
  envelope in the band below. Hours with no recording are drawn as explicit
  red-banded regions.
* **Click any lane to seek.** Hover and the *Point under cursor* panel resolves
  the absolute hour, the grid index, the source file, the cut position inside the
  original 2-minute recording, the event position, score and peak level.
* **Play any site** from its lane header or the *Sites* list. Switching site
  keeps the playhead on the same wall-clock hour, so you can A/B the same hour
  at three locations.
* A zoomed detail strip follows the playhead at 200 points/s, labelled with the
  current absolute hour and source file.
* Sliders for **excerpt length**, **gap** (the pulse), **speed**, sample rate and
  the **missing-hour policy**; *Render preview* rebuilds all three streams in
  ~3 s and reloads the viewer.
* **Export scope**: *active site only* (a single `coll`/`csv` for that site) or
  *all sites* (every site's file in one bundle), for `coll`, `cuepoints`, `csv`,
  `dict.json` and the Max notes.
* **Render preview** rebuilds all three streams and *waits for the new audio to
  be seekable* before reporting done, resuming at the same hour and play state.
  "done" means listening-ready, not merely that the files were written.
* **Limiter** switch and an adjustable ceiling, so you can A/B the levelling
  against the raw material by ear.
* **`play at`** — a live, pitch-preserving playback speed. Use this to skim;
  leave the render-time speed at 1.00× unless the file needs the change baked in.
* **`■ stop`** — a master stop. Pauses and rewinds to the start of the current
  excerpt, so pressing play again replays that excerpt rather than resuming
  midway. **Space** toggles the transport (ignored while typing in a control).

Absolute hours are shown in UTC and are derived from the same 1970-01-01 epoch
the index uses; the displayed hour is cross-checked against each recording's
filename by the test suite (5,558 slots, 0 mismatches).

### 4.4 Adjustable preview parameters

| parameter | default | effect |
|---|---|---|
| `cut_seconds` | 0.5 | excerpt length. Free to change at render time: the event is always re-placed to sit just inside the start, so short excerpts do not lose it. |
| `gap_seconds` | 0.05 | silence after each excerpt — this creates the pulse. **Negative overlaps** them. |
| `overlap_scale` | auto | level handling when overlapping (`auto` \| `unity` \| fixed gain) |
| `fade_seconds` | 0.004 | raised-cosine edge fade; the crossfade when overlapping (capped at cut/2) |
| `speed` | 1.0 | bake a speed change into the file (`--speed-mode pitch\|varispeed`) |
| `normalise` | off | optional per-excerpt peak normalisation |
| `out_sample_rate` | 16000 | preview rate; 16 kHz keeps the whole scan at 93 MB |
| `event_lead_s` | 0.010667 | pre-roll before the event when re-placing (`max(lead, 10 % of excerpt)`) |
| `limiter` | on | push peaks down so loud events cannot drown quiet ones |
| `limiter_ceiling_dbfs` | −12.0 | limiter ceiling; near 0 dBFS effectively disables it |

Total stream length is `n × (cut + gap)`: 5,558 × 0.55 s = **50.95 min**.

---

## 5. Export to Max/MSP

**One `coll` per site** (`points_moth-oculus.coll`, `points_moth-juvet.coll`,
`points_moth-kvarvet.coll`), plus a combined `points.coll` with a site-name
column. Each is **index-addressable**: event *i* is line *i*+1 (the file begins
with a `;` header line, as `coll` imports expect).

| column | meaning |
|---|---|
| 0 | event index within this site |
| 1 | **grid** — absolute hour index on the shared timeline |
| 2 | grid origin of this site's stream |
| 3 | **`stream_ms`** — offset into **this site's** preview WAV |
| 4 | **`cut_ms`** — cut position inside the source WAV |
| 5 | event position inside the source WAV, ms |
| 6 | excerpt duration, ms |
| 7 | score × 1000 |
| 8 | peak level, dBFS × 10 |
| 9 | 1 = passed the activity gate |
| 10 | `epoch_hour` — hours since 1970-01-01T00:00Z |
| 11 | **`noise`** — contamination score × 1000 (wind / clipping) |
| 12 | **`flags`** — wind×1 + clipped×2 + sustained×4 |
| 13 | **`substituted`** — 1 when the audio came from a different hour |
| 14 | **`origin_grid`** — grid hour the audio came from, or −1 |
| 15 | quoted absolute source path (a Max symbol) |

```
; mothscan cue points - site moth oculus - 2376 events cut=0.5s gap=0.05s
50 494296 494244 28600 54869 55045 500 3556 -361 0 494296 457 1 1 494297 "/Volumes/Solardisk/Harpefoss/moth oculus/20260522_170000.WAV"
                                                       ^noise ^flags ^sub ^origin
```
Row 50 is a substituted hour: its own hour (494296) was wind-dominated
(`noise 457`, `flags 1` = wind), so the audio and the path come from hour
494297 instead — `cut_ms 54869` is *that* hour's cut, not the original's 54869.
Filter to clean hours only with `column 11 <= 400 and column 12 == 0`.

* **Column 3** → play that site's preview: `[coll] → [line 3] → [sfplay~]`.
* **Columns 4 + 11** → play the original recordings: read the cut position and
  the path, `[open]` it on a second `[sfplay~]`, start at `cut_ms`.
* **Column 1 is the alignment key.** Two events from different sites with the
  same `grid` are the same wall-clock hour and sit at the same offset in their
  respective streams. Absolute instant = `epoch + grid × 3600 s`.
* **Column 9** → skip hours where nothing crossed the activity gate.
* **Column 7** → sort by strength to play only the best events of a site.
* **Columns 11–14** → wind/clip provenance. `13 = 1` means the excerpt was
  borrowed from `14`; `11` is how contaminated the hour it replaced was.

`max_notes.txt` in the export directory contains these recipes (including a
three-site aligned-playback recipe) as a Max-patch ready reference.

---

## 6. Layout

```
mothscan/
  mothscan/
    config.py        dataset facts + AnalysisConfig / PreviewConfig
    features.py      STFT, mel, novelty curve, envelopes, flatness
    scoring.py       the hybrid score, event-centred placement, 2-pass search
    pipeline.py      discovery, parallel analysis, resumable runs, retune
    index_store.py   SQLite + JSONL index, feature cache
    timeline.py      the shared hourly grid, slot mapping, missing-data policy
    noise.py         wind / clipping detection, penalties, substitution policy
    select.py        final point choice incl. nearby-hour substitution
    preview.py       per-site pulse-train renderer + binary browser assets
    export.py        per-site coll / cuepoints / csv / dict / Max notes
    webapp.py        FastAPI: multi-site viewer, metadata, render, export
    cli.py           scan / retune / stats / inspect / pick / preview / export / serve
  web/static/index.html    the whole viewer (no build step, no plotting dep)
  scripts/
    bench.py         per-file timing + score breakdown
    diagnose.py      spectrogram + score-curve + waveform figures
    characterise_noise.py  measure wind/clipping indicators over a sample
    calibrate_noise.py     indicator distributions, flags, worst-offender list
    validate_noise.py      labelled galleries to check the thresholds by eye
  tests/test_preview.py    unit tests: speed direction, pitch, limiter ceiling
    verify_ui.mjs    headless behavioural test of the viewer
    shoot.mjs        deterministic screenshot
```

### Environment

```bash
./mothscan.sh setup            # recommended: venv + deps + an import check
```

or by hand:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The wrapper finds the venv itself (`.venv`, then `../.venv`, then
`MOTHSCAN_VENV`) and sets `MPLCONFIGDIR` for you. Set it manually only if you
call the Python modules directly and matplotlib complains about `~/.matplotlib`
being unwritable (it is, on some machines):

```bash
export MPLCONFIGDIR="$PWD/../data/.mplcache"
```

---

## 7. If you want to go further

* **Tune for a target sound.** Use `diagnose.py` on hours you know contain what
  you want, and adjust `w_tonality` (birds/insects) or `w_flatness` +
  `w_transient` (rain/impacts) until the picks match your ear. The feature cache
  makes each iteration cost seconds.
* **More than one excerpt per hour.** `EventRecord.alternatives` already holds
  the top 5 distinct moments per file, ranked, with their own cut points,
  scores and levels — enough to build a longer or denser stream without
  re-analysis.
* **Longer deployments.** The grid is unbounded: a year of hourly files is 8,784
  slots per site, and because the pitch is a render parameter, raising
  `gap_seconds` compresses a longer span into the same stream length without
  changing any code. Sites that do not overlap in time simply have silent slots
  where the other was not deployed — exactly the `moth-kvarvet` case here.
* **True simultaneous playback in the viewer.** The streams are sample-locked
  across sites, so playing several at once with a per-site gain would give a
  live three-site comparison; the Max recipe in `max_notes.txt` (recipe C)
  already describes the patch.
* **New sites or years.** Add the directory name to `SITE_DIRS` (or pass
  `--sites`); a site split into `YYYYMMDD/` sub-folders is walked automatically.
* **Tune the wind thresholds to another deployment.** The absolute thresholds
  were calibrated against *this* dataset's distributions. Run
  `scripts/calibrate_noise.py` on the new data and move `wind_low100` /
  `wind_tilt_db_oct` onto its own gap. The physical logic (low-band dominance +
  steep tilt + a rail at 0 dBFS) transfers; the numbers should be re-earned.
* **A louder-but-fairer levelling option.** The limiter deliberately refuses to
  lift quiet material. If you would rather hear the quiet hours *and* keep the
  relative ordering, the natural extension is a slow per-hour gain ride (a
  compressor with a long release and a bounded makeup gain) rather than the
  instantaneous peak limiter — bounded so it cannot resurrect a silent hour.
* **Spectral-diversity term.** If a target sound is spectrally *complex* rather
  than merely novel, add a per-frame entropy term to `features.py` and give it a
  weight in `scoring.py`; the framework is already built around pluggable,
  individually normalised terms.
