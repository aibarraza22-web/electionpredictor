"""Poll-free forecast, chosen and sized by its own track record.

No polls, no poll-derived expert ratings. Every input is a hard, pre-election
fact that also existed for every past election, so the whole system can be
replayed on 2012-2024 exactly as it runs on 2026 (trained from 1990 on):

* **district/state partisanship** -- Cook PVI, i.e. how the place voted for
  President (``app.ingest.cook_pvi``), on the CURRENT lines, so redrawn 2026
  seats get a real baseline;
* **seat history** -- the last contested result on the same map;
* **statewide House lean** (``features.StateLean``);
* **incumbency** -- is the sitting member on the ballot (``app.incumbency``);
* **the national pattern** -- the president's party and midterm/presidential
  year, whose effect is fitted from certified results only.

Several SYSTEMS combine those facts in different ways. Every system is run
walk-forward -- trained only on cycles before the one it predicts -- so each
has a genuine out-of-sample record on every past race. Two uses of that
record were tested (research claim T-004):

1. **Which system to trust.** Picking a different system per seat from its
   last three elections never beat one system chosen on the chamber's
   hundreds of past close races (2016-24: House 94.4% vs 94.6% of races,
   Senate 88.6% vs 91.6%; per-state picks were worse still) -- three
   elections cannot separate skill from luck. The chamber-wide choice is
   stable (the same system every cycle from 2014 on), so the published
   system is the one with the best close-race record, re-chosen every run.
2. **How sure to be.** A seat's and a state's own past misses DO predict
   its future misses. Each race's uncertainty is a shared national term (what
   waves do to every seat) plus a local term shrunk from the seat's and
   state's own residuals: log loss improved in 11 of 12 held-out
   chamber-cycles vs one uniform uncertainty. The national floor is what
   stops a quiet local history from hiding wave risk.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from statistics import fmean, pstdev

from . import redistricting
from .domain import normal_cdf
from .features import PRESIDENT_PARTY, ResultLookup, StateLean, clip
from .incumbency import IncumbencyLookup
from .ingest import cook_pvi
from .model import ridge_fit

# President's party at each November election, facts of record. Extends
# features.PRESIDENT_PARTY (1998+) back to 1976 so the national pattern is
# fitted on every cycle of certified results.
PRESIDENT = {1976: "R", 1978: "D", 1980: "D", 1982: "R", 1984: "R", 1986: "R",
             1988: "R", 1990: "R", 1992: "R", 1994: "D", 1996: "D", **PRESIDENT_PARTY}

TRAIN_START = 1990          # first cycle of training rows
FIRST_SCORED_CYCLE = 2010   # first cycle whose systems all trained on PVI-era data
MIN_TRAINING_ROWS = 200
CLOSE_RACE = 20.0           # |actual margin| below this = a race worth calling
COMPETITIVE_PREDICTION = 25.0
Y_CLIP = 40.0               # blowouts carry no information about close races
UNCONTESTED = 99.0          # |margin| at/above this: one major party absent
L2 = 4.0
LOCAL_ELECTIONS = 3         # "the past three elections here"
SEAT_SHRINK = 3.0           # pseudo-observations pulling a seat toward its state
STATE_SHRINK = 20.0         # ... and a state toward the chamber
MIN_SIGMA = 3.0
# The September 2012 PVI revision still described the pre-2012 districts.
PVI_ON_OLD_LINES = {2012}

SYSTEMS: dict[str, dict] = {
    "persistence": {"features": ["prior", "env"],
                    "label": "Seat history",
                    "about": "the seat votes like it did last time on these lines, "
                             "shifted by the usual swing for this kind of year"},
    "pvi": {"features": ["pvi", "env"],
            "label": "Presidential lean",
            "about": "how the district voted for President relative to the nation"},
    "pvi_inc": {"features": ["pvi", "inc", "env"],
                "label": "Presidential lean + incumbency",
                "about": "presidential lean, plus whether the sitting member is on "
                         "the ballot"},
    "core": {"features": ["stale", "lean", "env"],
             "label": "Last result + state lean",
             "about": "last result on any lines plus the statewide House lean "
                      "(the previous model with its polls removed)"},
    "full": {"features": ["prior", "pvi", "lean", "inc", "open_x_prior", "env"],
             "label": "Everything, one regression",
             "about": "every poll-free input in a single fitted equation"},
}
ENSEMBLE = "ensemble"
ENSEMBLE_LABEL = "Average of all five systems"
CANDIDATES = list(SYSTEMS) + [ENSEMBLE]

# Feature -> human group, for the additive "why this forecast" breakdown.
GROUPS = {"intercept": "baseline", "prior": "seat history", "has_prior": "seat history",
          "stale": "seat history", "has_stale": "seat history",
          "pvi": "presidential lean", "has_pvi": "presidential lean",
          "lean": "statewide House lean", "has_lean": "statewide House lean",
          "inc": "incumbency", "open_x_prior": "seat history",
          "env": "national environment", "mid": "national environment"}


def label(system: str) -> str:
    return ENSEMBLE_LABEL if system == ENSEMBLE else SYSTEMS[system]["label"]


def environment(cycle: int) -> tuple[float, float]:
    """(+1 when the out-party is D, same but midterms only)."""
    sign = 1.0 if PRESIDENT[cycle] == "R" else -1.0
    return sign, sign if cycle % 4 == 2 else 0.0


@dataclass
class SeatRow:
    seat_key: str
    chamber: str
    state: str
    district: str | None
    cycle: int
    y: float | None
    prior: float | None = None
    prior_cycle: int | None = None
    stale_prior: float | None = None
    pvi: float | None = None
    lean: float | None = None
    inc: float | None = None
    env: float = 0.0
    mid: float = 0.0
    redrawn: bool = False
    extra: dict = field(default_factory=dict)


def vector(row: SeatRow, features: list[str]) -> tuple[list[str], list[float]]:
    names, x = ["intercept"], [1.0]
    has_prior = row.prior is not None
    inc = row.inc if row.inc is not None else 0.0
    if "prior" in features:
        names += ["prior", "has_prior"]
        x += [clip(row.prior) if has_prior else 0.0, float(has_prior)]
    if "stale" in features:
        names += ["stale", "has_stale"]
        x += [clip(row.stale_prior) if row.stale_prior is not None else 0.0,
              float(row.stale_prior is not None)]
    if "pvi" in features:
        names += ["pvi", "has_pvi"]
        x += [row.pvi if row.pvi is not None else 0.0, float(row.pvi is not None)]
    if "lean" in features:
        names += ["lean", "has_lean"]
        x += [clip(row.lean) if row.lean is not None else 0.0, float(row.lean is not None)]
    if "inc" in features:
        names.append("inc")
        x.append(inc)
    if "open_x_prior" in features:  # an open seat inherits less of its last margin
        names.append("open_x_prior")
        x.append((clip(row.prior) if has_prior else 0.0) * (1.0 if inc == 0 else 0.0))
    if "env" in features:
        names += ["env", "mid"]
        x += [row.env, row.mid]
    return names, x


class Inputs:
    """Every poll-free fact the systems read, keyed for any cycle."""

    def __init__(self, results: ResultLookup, incumbency: IncumbencyLookup | None = None,
                 pvi: tuple[dict, dict] | None = None):
        self.results = results
        self.state_lean = StateLean(results)
        self.incumbency = incumbency or IncumbencyLookup()
        self.pvi_district, self.pvi_state = pvi or cook_pvi.load_vintages()

    def pvi(self, chamber: str, cycle: int, state: str, district: str | None) -> float | None:
        if chamber == "house":
            if cycle in PVI_ON_OLD_LINES:
                return None
            try:
                number = max(int(district or 1), 1)
            except ValueError:
                return None
            return self.pvi_district.get((cycle, state, number))
        return self.pvi_state.get((cycle, state))

    def house_prior(self, cycle: int, seat_key: str, state: str) -> tuple[float | None, int | None]:
        """Most recent CONTESTED result for this seat on the SAME map."""
        for back in (2, 4):
            prior_cycle = cycle - back
            if redistricting.map_changed(state, prior_cycle, cycle):
                return None, None
            margin = self.results.margin(prior_cycle, seat_key)
            if margin is not None and abs(margin) < UNCONTESTED:
                return margin, prior_cycle
        return None, None

    def row(self, chamber: str, cycle: int, seat_key: str, state: str,
            district: str | None, y: float | None = None,
            holder_party: str | None = None) -> SeatRow:
        stale, stale_cycle = self.results.prior(cycle, seat_key, chamber)
        if chamber == "house":
            prior, prior_cycle = self.house_prior(cycle, seat_key, state)
            redrawn = stale is not None and prior is None and \
                redistricting.map_changed(state, stale_cycle, cycle)
        else:
            prior, prior_cycle = stale, stale_cycle
            if prior is not None and abs(prior) >= UNCONTESTED:
                prior, prior_cycle = None, None
            stale = prior  # statewide races: no map to go stale, only uncontested gaps
            redrawn = False
        inc = self.incumbency.value(cycle, seat_key)
        env, mid = environment(cycle)
        return SeatRow(
            seat_key=seat_key, chamber=chamber, state=state, district=district,
            cycle=cycle, y=y, prior=prior, prior_cycle=prior_cycle, stale_prior=stale,
            pvi=self.pvi(chamber, cycle, state, district),
            lean=self.state_lean.lean(state, cycle)[0], inc=inc, env=env, mid=mid,
            redrawn=redrawn, extra={"holder_party": holder_party})

    def history(self, chamber: str, first: int = TRAIN_START, last: int = 2024) -> dict[int, list[SeatRow]]:
        rows: dict[int, list[SeatRow]] = {}
        for cycle in self.results.cycles(chamber):
            if cycle < first or cycle > last or cycle % 2 or cycle not in PRESIDENT:
                continue
            seats = []
            for result in self.results.seats(cycle, chamber):
                if abs(result["dem_margin"]) >= UNCONTESTED:
                    continue  # not a two-party contest: nothing to learn
                seats.append(self.row(chamber, cycle, result["seat_key"], result["state"],
                                      result.get("district"), y=result["dem_margin"]))
            if seats:
                rows[cycle] = seats
        return rows


class System:
    """One fitted ridge system (cycle-level terms penalised by rows-per-cycle,
    as in ``model.MarginModel``, so ~15 national data points are not treated
    as thousands)."""

    def __init__(self, name: str):
        self.name = name
        self.features = SYSTEMS[name]["features"]
        self.weights: list[float] = []
        self.names: list[str] = []

    def fit(self, rows: list[SeatRow]) -> "System":
        xs = []
        for row in rows:
            self.names, x = vector(row, self.features)
            xs.append(x)
        ys = [clip(row.y, Y_CLIP) for row in rows]
        per_cycle = max(1.0, len(rows) / max(1, len({r.cycle for r in rows})))
        penalties = [L2 * per_cycle if name in ("env", "mid") else L2 for name in self.names]
        self.weights = ridge_fit(xs, ys, penalties)
        return self

    def predict(self, row: SeatRow) -> float:
        _, x = vector(row, self.features)
        return sum(w * v for w, v in zip(self.weights, x))

    def contributions(self, row: SeatRow) -> dict[str, float]:
        names, x = vector(row, self.features)
        grouped: dict[str, float] = defaultdict(float)
        for name, w, v in zip(names, self.weights, x):
            grouped[GROUPS[name]] += w * v
        return dict(grouped)


def fit_systems(rows: list[SeatRow]) -> dict[str, System]:
    """Each system is trained only on races where its defining input exists
    (the same rule ``usable`` applies at prediction time). Training the PVI
    systems on 1990-2006 rows that have no PVI let incumbency stand in for
    partisanship: its coefficient inflated to 22.6 margin points and the
    model rated the country's closest House seats as Likely R. Trained on
    their own inputs it is 11.7, and held-out log loss improved in both
    chambers (House 0.197 -> 0.169, Senate 0.233 -> 0.224, 2014-24).
    Up-weighting ex-ante competitive races was also tried and NOT adopted
    (better in 3 of 6 House cycles, worse in the Senate)."""
    systems = {}
    for name in SYSTEMS:
        own = [r for r in rows if trains_on(name, r)]
        systems[name] = System(name).fit(own or rows)
    return systems


def predict_all(systems: dict[str, System], row: SeatRow) -> dict[str, float]:
    out = {name: system.predict(row) for name, system in systems.items()}
    out[ENSEMBLE] = fmean(out[name] for name in SYSTEMS)
    return out


@dataclass
class Scored:
    """One out-of-sample prediction set for a past race."""
    cycle: int
    seat_key: str
    state: str
    y: float
    preds: dict[str, float]
    pvi: float | None = None
    prior: float | None = None


def walk_forward(history: dict[int, list[SeatRow]]) -> dict[tuple[int, str], Scored]:
    """Out-of-sample predictions of every candidate for every eligible cycle."""
    scored: dict[tuple[int, str], Scored] = {}
    cycles = sorted(history)
    for target in cycles:
        training = [row for cycle in cycles if cycle < target for row in history[cycle]]
        if len(training) < MIN_TRAINING_ROWS:
            continue
        assert all(row.cycle < target for row in training), "future cycle leaked into training"
        systems = fit_systems(training)
        for row in history[target]:
            scored[(target, row.seat_key)] = Scored(
                target, row.seat_key, row.state, row.y, predict_all(systems, row),
                pvi=row.pvi, prior=row.prior)
    return scored


def _close_record(records: list[Scored], system: str) -> tuple[float, float, int]:
    close = [s for s in records if abs(s.y) < CLOSE_RACE]
    if not close:
        return 0.0, float("inf"), 0
    accuracy = fmean((s.preds[system] > 0) == (s.y > 0) for s in close)
    mse = fmean((clip(s.y, Y_CLIP) - s.preds[system]) ** 2 for s in close)
    return accuracy, mse, len(close)


def rank_systems(records: list[Scored]) -> list[str]:
    """Candidates ordered by close-race winner record (ties: lower squared
    error). With no record yet, the ensemble first -- no choice is made."""
    if not records:
        return [ENSEMBLE] + list(SYSTEMS)
    return sorted(CANDIDATES, key=lambda c: (-round(_close_record(records, c)[0], 3),
                                              _close_record(records, c)[1]))


# A system is only used on a race whose defining input exists. Without its
# PVI a presidential-lean system is a constant, and without a same-map result
# persistence is too; the combined systems degrade gracefully instead. Found
# replaying 2012: the pick made from 2010 alone was the PVI system, and PVI on
# the new 2012 lines was not published before that election -- 47% accuracy.
REQUIRES = {"pvi": "pvi", "pvi_inc": "pvi", "persistence": "prior"}


def trains_on(system: str, row) -> bool:
    """Rows a system learns from: those carrying every input it relies on."""
    needs = {"pvi": ("pvi",), "pvi_inc": ("pvi",), "full": ("pvi",),
             "persistence": ("prior",)}.get(system, ())
    return all(getattr(row, need) is not None for need in needs)


def usable(system: str, row) -> bool:
    need = REQUIRES.get(system)
    return need is None or getattr(row, need) is not None


def pick_for(ranking: list[str], row) -> str:
    return next(c for c in ranking if usable(c, row))


class Uncertainty:
    """sigma^2 = national^2 + local^2, from a system's out-of-sample record.

    national: SD of each past cycle's mean error on competitive seats -- the
    part of the miss every seat shares (waves). local: the seat's own and its
    state's idiosyncratic errors over the last LOCAL_ELECTIONS cycles, each
    shrunk toward the level above it by SEAT_SHRINK / STATE_SHRINK
    pseudo-observations."""

    def __init__(self, records: list[Scored], system: str, chamber: str):
        self.system = system
        self.chamber = chamber
        competitive = [s for s in records if abs(s.preds[system]) < COMPETITIVE_PREDICTION]
        by_cycle: dict[int, list[float]] = defaultdict(list)
        for s in competitive:
            by_cycle[s.cycle].append(clip(s.y, Y_CLIP) - s.preds[system])
        self.cycle_mean = {c: fmean(v) for c, v in by_cycle.items()}
        self.national = pstdev(self.cycle_mean.values()) if len(self.cycle_mean) > 1 else 0.0
        self.idio: dict[tuple[int, str], float] = {}
        self.by_state: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for s in competitive:
            error = clip(s.y, Y_CLIP) - s.preds[system] - self.cycle_mean[s.cycle]
            self.idio[(s.cycle, s.seat_key)] = error
            self.by_state[s.state].append((s.cycle, error))
        self.global_idio = pstdev(self.idio.values()) if len(self.idio) > 1 else 8.0

    def local(self, cycle: int, seat_key: str, state: str) -> dict:
        window = [cycle - 2 * k for k in range(1, LOCAL_ELECTIONS + 1)]
        variance = self.global_idio ** 2
        state_errors = [e for c, e in self.by_state.get(state, ()) if window[-1] <= c < cycle]
        if state_errors:
            variance = ((sum(e * e for e in state_errors) + STATE_SHRINK * variance)
                        / (len(state_errors) + STATE_SHRINK))
        seat_errors = [self.idio[(c, seat_key)] for c in window if (c, seat_key) in self.idio
                       and (self.chamber == "senate"
                            or not redistricting.map_changed(state, c, cycle))]
        if seat_errors:
            variance = ((sum(e * e for e in seat_errors) + SEAT_SHRINK * variance)
                        / (len(seat_errors) + SEAT_SHRINK))
        return {"national": round(self.national, 3), "local": round(sqrt(variance), 3),
                "sigma": max(MIN_SIGMA, sqrt(self.national ** 2 + variance)),
                "state_errors": len(state_errors), "seat_errors": len(seat_errors)}


def probability(mean: float, sigma: float, calibration: tuple | None = None) -> float:
    """Win probability from a margin and its sigma; with ``calibration``
    (a, b) it is the logistic read of z = mean/sigma fitted on past outcomes."""
    if calibration:
        from math import exp
        a, b = calibration
        z = max(-30.0, min(30.0, a + b * mean / sigma))
        return min(0.995, max(0.005, 1.0 / (1.0 + exp(-z))))
    return min(0.995, max(0.005, normal_cdf(mean / sigma)))


def fit_calibration(rows: list[tuple], min_rows: int = 50) -> tuple | None:
    """Slope-only logistic P(D win) = sigmoid(b*z), z = margin/sigma, fitted
    by Newton's method on past out-of-sample (margin, sigma, actual) rows.

    No intercept on purpose. A free intercept fitted the average past
    national miss (Democrats beat the fundamentals in 2016/18/22/24, fell
    short in 2014/20) and would have tilted every 2026 toss-up toward the
    Democrats on four cycles' say-so; slope-only scores the same held-out log
    loss (House 0.1692 vs 0.1691, Senate 0.2076 vs 0.2067) without that
    guess. The national miss is the simulation's national-shock term."""
    from math import exp
    zs = [m / s for m, s, _y, *_ in rows]
    won = [1 if y > 0 else 0 for _m, _s, y, *_ in rows]
    if len(zs) < min_rows or len(set(won)) < 2:
        return None
    b = 1.6
    for _ in range(100):
        grad = hess = 0.0
        for z, w in zip(zs, won):
            p = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, b * z))))
            grad += (p - w) * z
            hess += p * (1.0 - p) * z * z
        if hess < 1e-9:
            break
        step = grad / hess
        b -= step
        if abs(step) < 1e-10:
            break
    return (0.0, b) if b > 0 else None


def _log_loss(rows: list[tuple]) -> float:
    from math import log
    total = 0.0
    for p, y in rows:
        won = 1 if y > 0 else 0
        total -= won * log(p) + (1 - won) * log(1 - p)
    return total / len(rows)


def seat_record(records: dict[tuple[int, str], Scored], row: SeatRow,
                system: str) -> list[dict]:
    """How ``system`` did in "the past three elections here", out of sample:
    this House seat's last three results on the same map, or this state's
    last three Senate races (either seat -- a Senate "area" is the state)."""
    if row.chamber == "house":
        past = []
        for back in range(1, 8):
            cycle = row.cycle - 2 * back
            if redistricting.map_changed(row.state, cycle, row.cycle):
                break
            if (cycle, row.seat_key) in records:
                past.append(records[(cycle, row.seat_key)])
    else:
        prefix = f"senate-{row.state}"
        past = sorted((s for (c, key), s in records.items()
                       if c < row.cycle and (key == prefix or key.startswith(prefix + "-"))),
                      key=lambda s: -s.cycle)
    out = []
    for scored in past:
        if scored.cycle < FIRST_SCORED_CYCLE:
            continue
        predicted = scored.preds[system]
        out.append({"cycle": scored.cycle, "seat_key": scored.seat_key,
                    "predicted_margin": round(predicted, 1),
                    "actual_margin": round(scored.y, 1),
                    "called_correctly": (predicted > 0) == (scored.y > 0),
                    "every_system_called_correctly": {
                        c: (scored.preds[c] > 0) == (scored.y > 0) for c in CANDIDATES}})
        if len(out) >= LOCAL_ELECTIONS:
            break
    return sorted(out, key=lambda item: item["cycle"])


def grade(row: SeatRow, record: list[dict]) -> str:
    """Evidence present for this seat, poll-free: partisanship on the current
    lines (2), same-map results (up to 2), known ballot status (1), and a
    local track record of 2+ elections to size its uncertainty (1)."""
    score = (2 * int(row.pvi is not None) + int(row.prior is not None)
             + int(row.prior is not None and len(record) >= 2)
             + int(row.inc is not None) + int(len(record) >= 2))
    return "A" if score >= 6 else "B" if score >= 4 else "C" if score >= 2 else "D" if score >= 1 else "F"


def metrics(rows: list[tuple]) -> dict:
    """rows: (predicted margin, sigma, actual margin[, win probability]) --
    the probability defaults to the uncalibrated normal reading."""
    from math import log
    n = len(rows)
    if not n:
        return {}
    pairs = [(r[0], r[1], r[2]) for r in rows]
    probs = [r[3] if len(r) > 3 else probability(r[0], r[1]) for r in rows]
    won = [1 if y > 0 else 0 for _, _, y in pairs]
    close = [(p, y) for p, _, y in pairs if abs(y) < CLOSE_RACE]
    inside80 = [abs(y - p) <= 1.282 * s for p, s, y in pairs]
    inside95 = [abs(y - p) <= 1.960 * s for p, s, y in pairs]
    return {
        "n_races": n,
        "winner_accuracy": round(fmean((p > 0) == (y > 0) for p, _, y in pairs), 4),
        "close_race_accuracy": round(fmean((p > 0) == (y > 0) for p, y in close), 4) if close else None,
        "n_close_races": len(close),
        "tossup_race_accuracy": (round(fmean((p > 0) == (y > 0) for p, y in close if abs(y) < 10), 4)
                                 if any(abs(y) < 10 for _, y in close) else None),
        "n_tossup_races": sum(1 for _, y in close if abs(y) < 10),
        "brier": round(fmean((q - w) ** 2 for q, w in zip(probs, won)), 4),
        "log_loss": round(-fmean(w * log(q) + (1 - w) * log(1 - q) for q, w in zip(probs, won)), 4),
        "margin_mae": round(fmean(abs(clip(y, Y_CLIP) - p) for p, _, y in pairs), 3),
        "coverage80": round(fmean(inside80), 4),
        "coverage95": round(fmean(inside95), 4),
        "seat_error": sum(p > 0 for p, _, _ in pairs) - sum(y > 0 for _, _, y in pairs),
    }


class TrackRecordModel:
    """Fit once per chamber; predicts any SeatRow for the target cycle."""

    def __init__(self, chamber: str):
        self.chamber = chamber
        self.ranking: list[str] = [ENSEMBLE]
        self.systems: dict[str, System] = {}
        self.records: dict[tuple[int, str], Scored] = {}
        self.uncertainty: dict[str, Uncertainty] = {}
        self.scoreboard: dict = {}
        self.backtest: dict = {}
        # (candidate, cycle, seat_key, predicted margin, sigma, actual margin,
        # win probability) for every replayed held-out race; "published" is
        # what the full procedure would have published.
        self.backtest_rows: list[tuple] = []
        # Win-probability calibration, adopted per chamber only when the
        # walk-forward replay says it helps (see _decide_calibration).
        self.calibration: tuple | None = None
        self.calibration_decision: dict = {}

    @property
    def system(self) -> str:
        """The chamber's published system (first in the track-record ranking)."""
        return self.ranking[0]

    def fit(self, inputs: Inputs, target_cycle: int) -> "TrackRecordModel":
        history = inputs.history(self.chamber, last=target_cycle - 2)
        self.records = walk_forward(history)
        scored = [s for s in self.records.values() if s.cycle >= FIRST_SCORED_CYCLE]
        self.ranking = rank_systems(scored)
        self.uncertainty = {c: Uncertainty(list(self.records.values()), c, self.chamber)
                            for c in CANDIDATES}
        self.systems = fit_systems([row for rows in history.values() for row in rows])
        self.scoreboard = {c: {"label": label(c), **metrics_close(scored, c)} for c in CANDIDATES}
        self.backtest = self._honest_backtest()
        return self

    def _honest_backtest(self) -> dict:
        """Replays the WHOLE procedure per held-out cycle -- ranking, per-race
        fallback and uncertainty all use only cycles before it -- so the
        stored report card is what this model would actually have published."""
        by_cycle: dict[int, list] = {}
        per_system: dict[str, list] = defaultdict(list)
        chosen: dict[int, str] = {}
        seat_order: dict[int, list] = defaultdict(list)
        self.backtest_rows = []
        for target in sorted({s.cycle for s in self.records.values()}):
            if target < FIRST_SCORED_CYCLE + 2:
                continue
            past = [s for s in self.records.values() if s.cycle < target]
            ranking = rank_systems([s for s in past if s.cycle >= FIRST_SCORED_CYCLE])
            chosen[target] = ranking[0]
            unc = {c: Uncertainty(past, c, self.chamber) for c in CANDIDATES}
            rows = []
            for s in self.records.values():
                if s.cycle != target:
                    continue
                system = pick_for(ranking, s)
                sigma = unc[system].local(target, s.seat_key, s.state)["sigma"]
                rows.append((s.preds[system], sigma, s.y))
                seat_order[target].append((s.seat_key, system))
                for c in CANDIDATES:
                    c_sigma = unc[c].local(target, s.seat_key, s.state)["sigma"]
                    per_system[c].append((s.preds[c], c_sigma, s.y))
                    self.backtest_rows.append((c, target, s.seat_key, s.preds[c], c_sigma, s.y,
                                               probability(s.preds[c], c_sigma)))
            by_cycle[target] = rows
        by_cycle = self._decide_calibration(by_cycle)
        for cycle, rows in by_cycle.items():
            for (seat_key, _), row in zip(seat_order[cycle], rows):
                self.backtest_rows.append(("published", cycle, seat_key, *row))
        everything = [r for rows in by_cycle.values() for r in rows]
        return {"cycles": sorted(by_cycle), "chosen_system_by_cycle": chosen,
                "summary": metrics(everything),
                "by_cycle": {str(c): metrics(rows) for c, rows in by_cycle.items()},
                "national_error_sigma_pts": round(self.uncertainty[self.system].national, 3),
                "calibration": self.calibration_decision,
                "per_system": {c: metrics(v) for c, v in per_system.items()}}

    def _decide_calibration(self, by_cycle: dict[int, list]) -> dict[int, list]:
        """Should win probabilities be calibrated on past outcomes?

        Replayed walk-forward: each held-out cycle is calibrated on the
        published rows of the cycles before it. Adopted only if that beats the
        raw normal reading on mean log loss AND in most of those cycles, so
        each chamber gets what its own record supports (at 2026.21 both pass:
        House 0.1705 -> 0.1692, better in 4 of 6 cycles; Senate 0.2500 ->
        0.2076, 4 of 5). Returns rows carrying the probability the decided
        procedure would have published."""
        cycles = sorted(by_cycle)
        raw, calibrated, fitted_on = {}, {}, {}
        for cycle in cycles:
            past = [r for c in cycles if c < cycle for r in by_cycle[c]]
            fit = fit_calibration(past)
            raw[cycle] = [(probability(m, s), y) for m, s, y in by_cycle[cycle]]
            if fit:
                calibrated[cycle] = [(probability(m, s, fit), y) for m, s, y in by_cycle[cycle]]
                fitted_on[cycle] = len(past)
        judged = sorted(calibrated)
        better = [c for c in judged if _log_loss(calibrated[c]) < _log_loss(raw[c])]
        adopt = bool(judged) and (
            fmean(_log_loss(calibrated[c]) for c in judged) < fmean(_log_loss(raw[c]) for c in judged)
            and len(better) > len(judged) / 2)
        everything = [r for c in cycles for r in by_cycle[c]]
        self.calibration = fit_calibration(everything) if adopt else None
        self.calibration_decision = {
            "adopted": adopt, "cycles_judged": judged,
            "cycles_better": better,
            "log_loss_raw": {str(c): round(_log_loss(raw[c]), 4) for c in judged},
            "log_loss_calibrated": {str(c): round(_log_loss(calibrated[c]), 4) for c in judged},
            "fit": [round(v, 4) for v in self.calibration] if self.calibration else None,
            "rule": "adopt only if better on mean log loss AND in most held-out cycles"}
        out = {}
        for cycle in cycles:
            probs = calibrated[cycle] if adopt and cycle in calibrated else raw[cycle]
            out[cycle] = [(m, s, y, p) for (m, s, y), (p, _y) in zip(by_cycle[cycle], probs)]
        return out

    def predict(self, row: SeatRow) -> dict:
        preds = predict_all(self.systems, row)
        system = pick_for(self.ranking, row)
        mean = preds[system]
        unc = self.uncertainty[system].local(row.cycle, row.seat_key, row.state)
        sigma = unc["sigma"]
        if system == ENSEMBLE:
            groups: dict[str, float] = defaultdict(float)
            for fitted in self.systems.values():
                for group, value in fitted.contributions(row).items():
                    groups[group] += value / len(self.systems)
        else:
            groups = self.systems[system].contributions(row)
        record = seat_record(self.records, row, system)
        return {"system": system, "mean": mean, "sigma": sigma,
                "probability": probability(mean, sigma, self.calibration), "systems": preds,
                "contributions": dict(groups), "uncertainty": unc,
                "seat_record": record, "grade": grade(row, record)}


def metrics_close(records: list[Scored], system: str) -> dict:
    """JSON-safe close-race record (None, not inf, when there is no record)."""
    accuracy, mse, n = _close_record(records, system)
    return {"close_race_accuracy": round(accuracy, 4) if n else None,
            "close_race_mse": round(mse, 2) if n else None, "n_close_races": n}
