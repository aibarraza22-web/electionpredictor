"""Expert race ratings: parsing, consensus, vintage safety, and the overlay."""
import pytest

from app.domain import quality_grade
from app.ingest import race_ratings
from app.ratings import (RatingLookup, RatingOverlay, is_unanimously_safe,
                         overlay_consensus, rating_evidence_points)
from app.redistricting import NET_DEM_SEAT_SHIFT, current_map_cycle, prior_is_stale


NATIONAL_HOUSE_PAGE = """
Some prose about the ratings.
{| class="wikitable sortable"
|- valign=bottom
! [[List of United States congressional districts|District]]
! [[Cook Partisan Voting Index|CPVI]]<ref>{{cite web |url=http://x}}</ref>
! Incumbent
! Last result<ref name="r">{{cite web
|url=http://y}}</ref>
! [[Cook Political Report|Cook]]<br />{{small|Aug. 25,<br />2026}}<ref name="c">{{cite web
|url=http://z}}</ref>
! [[Sabato's Crystal Ball|Sabato]]<br />{{small|Aug. 26,<br />2026}}
! [[Decision Desk HQ|DDHQ]]{{efn|A footnote instead of a date.}}
! Result
|-
!{{ushr|AL|2|X}}
|{{shading PVI|R|7}}
| {{Party shading/Democratic}} |{{sortname|Shomari|Figures}}
| {{Party shading/Democratic}} data-sort-value="-54.6" |54.6% D
|{{USRaceRating|Likely|R|Flip}}
|{{USRaceRating|Lean|R|Flip}}
|{{USRaceRating|Tossup}}
|
|-
!{{Ushr|AK|AL|X}}
|{{Shading PVI|R|6}}
| {{Party shading/Republican}} |{{sortname|Nick|Begich III}}<br />{{Small|(retiring)}}
| {{Party shading/Republican}} data-sort-value="51.3" |51.3% R
|{{USRaceRating|Safe|R}}
|{{USRaceRating|Solid|R}}
|{{USRaceRating|Safe|R}}
|
|}
"""

STATE_HOUSE_PAGE = """
== District 1 ==
====Predictions====
{| class="wikitable"
!Source
!Ranking
!As of
|-
| align=left | [[Decision Desk HQ]]<ref name="d">{{cite web
|title=x |url=http://d}}</ref>
| {{USRaceRating|Safe|D|Flip}}
|July 14, 2026
|-
| style="text-align:left" | [[Inside Elections]]
| {{USRaceRating|Solid|D|Flip}}
|November 6, 2025
|}

== District 2 ==
====Predictions====
{| class="wikitable"
!Source
!Ranking
!As of
|-
| align=left | [[The Cook Political Report]]
| {{USRaceRating|Tilt|R}}
|August 3, 2026
|}
"""


def test_parse_rating_ladder_and_sign():
    assert race_ratings.parse_rating("Tossup") == ("Tossup", 0.0)
    assert race_ratings.parse_rating("Lean|R|Flip") == ("Lean R", -2.0)
    assert race_ratings.parse_rating("Likely|D") == ("Likely D", 3.0)
    assert race_ratings.parse_rating("Solid|R") == ("Solid R", -4.0)
    # Safe and Solid are the same rung of the ladder, spelled differently by
    # different raters; both must score identically or the consensus of a
    # unanimous field would depend on vocabulary.
    assert race_ratings.parse_rating("Safe|R")[1] == race_ratings.parse_rating("Solid|R")[1]
    assert race_ratings.parse_rating("Nonsense|D") is None


def test_national_page_parse_attributes_every_rater():
    ratings, contexts, stats = race_ratings.parse(
        NATIONAL_HOUSE_PAGE, 2026, "house", "http://page", "2026-08-31T00:00:00Z")
    assert stats["rater_columns"] == 3          # the trailing Result column is not a rater
    assert stats["unattributed_rows"] == 0
    by_seat = {}
    for row in ratings:
        by_seat.setdefault(row["seat_key"], {})[row["rater"]] = row
    # At-large districts normalize to 01, matching house_seat_key elsewhere.
    assert set(by_seat) == {"house-AL-02", "house-AK-01"}
    assert by_seat["house-AL-02"]["Cook Political Report"]["score"] == -3.0
    assert by_seat["house-AL-02"]["Cook Political Report"]["rating_date"] == "2026-08-25"
    # A rater whose header carries a footnote instead of a date inherits the
    # latest date on the page rather than being dropped or guessed forward.
    # The national page's "DDHQ" and a state page's "Decision Desk HQ" are one
    # organisation; canonical names stop the consensus counting it twice.
    assert by_seat["house-AL-02"]["Decision Desk HQ"]["rating_date"] == "2026-08-26"
    context = {c["seat_key"]: c for c in contexts}
    assert context["house-AL-02"]["cook_pvi"] == -7.0
    assert context["house-AK-01"]["incumbent_retiring"] is True
    assert context["house-AL-02"]["incumbent_retiring"] is False


def test_state_page_parse_dates_and_attributes_each_row():
    ratings, stats = race_ratings.parse_state_page(
        STATE_HOUSE_PAGE, 2026, "CA", "http://state", "2026-08-31T00:00:00Z")
    assert stats == {"districts_with_ratings": 2, "rating_rows": 3, "undated_rows": 0}
    first = {r["rater"]: r for r in ratings if r["seat_key"] == "house-CA-01"}
    # Refs span lines; if they are not stripped first the row's leading cell is
    # split and every rater collapses to one name, which the uniqueness key
    # would then silently dedupe down to a single rating.
    assert set(first) == {"Decision Desk HQ", "Inside Elections"}
    assert first["Decision Desk HQ"]["rating_date"] == "2026-07-14"
    assert first["Inside Elections"]["rating_date"] == "2025-11-06"
    second = [r for r in ratings if r["seat_key"] == "house-CA-02"]
    assert second[0]["score"] == -1.0
    assert second[0]["rater"] == "Cook Political Report"


def test_rater_names_are_canonical_across_page_types():
    assert race_ratings.canonical_rater("DDHQ") == "Decision Desk HQ"
    assert race_ratings.canonical_rater("Decision Desk HQ") == "Decision Desk HQ"
    assert race_ratings.canonical_rater("Econ.") == "The Economist"
    assert race_ratings.canonical_rater("Sabato") == "Sabato's Crystal Ball"
    # An unknown rater keeps its own name rather than being dropped.
    assert race_ratings.canonical_rater("Some New Outfit") == "Some New Outfit"


def test_senate_row_state_matching_is_longest_name_first():
    block = ('! [[2026 United States Senate election in West Virginia|West Virginia]]\n'
             '| {{USRaceRating|Solid|R}}')
    assert race_ratings._senate_seat(block)[2] == "senate-WV"
    special = ('! [[2026 United States Senate special election in Florida|Florida (special)]]\n'
               '| {{USRaceRating|Lean|R}}')
    assert race_ratings._senate_seat(special)[2] == "senate-FL-special"


def _rows(*specs):
    return [{"cycle": 2026, "seat_key": "house-XX-01", "rater": rater,
             "rating": label, "score": score, "rating_date": day,
             "source_url": "http://x"}
            for rater, label, score, day in specs]


def test_consensus_uses_each_raters_latest_rating_only():
    lookup = RatingLookup(_rows(
        ("Cook", "Lean R", -2.0, "2026-03-01"),
        ("Cook", "Tossup", 0.0, "2026-08-01"),     # Cook moved the race
        ("Sabato", "Tossup", 0.0, "2026-08-02"),
    ))
    summary = lookup.consensus(2026, "house-XX-01", as_of="2026-08-31")
    assert summary["n_raters"] == 2
    assert summary["consensus"] == 0.0             # not dragged by Cook's old call


def test_consensus_respects_the_as_of_cutoff():
    lookup = RatingLookup(_rows(
        ("Cook", "Lean R", -2.0, "2026-03-01"),
        ("Sabato", "Safe D", 4.0, "2026-09-15"),   # published after the cutoff
    ))
    summary = lookup.consensus(2026, "house-XX-01", as_of="2026-08-31")
    assert summary["n_raters"] == 1
    assert summary["consensus"] == -2.0
    assert lookup.consensus(2026, "house-XX-01", as_of="2026-01-01") is None


def test_rating_evidence_points_and_data_grade():
    fresh = {"n_raters": 7, "age_days": 10}
    thin = {"n_raters": 2, "age_days": 10}
    stale = {"n_raters": 7, "age_days": 400}
    assert rating_evidence_points(fresh) == 2
    assert rating_evidence_points(thin) == 1
    assert rating_evidence_points(stale) == 0
    assert rating_evidence_points(None) == 0
    # An unpolled race with current finance, a known candidate, certain
    # boundaries and a real expert consensus reaches B on evidence alone --
    # this is what takes competitive races off grade C.
    assert quality_grade(0, None, True, True, True, rating_points=2) == "B"
    assert quality_grade(0, None, True, True, True, rating_points=0) == "C"
    assert quality_grade(1, 5, True, True, True, rating_points=2) == "A"


def test_unanimous_safe_detection_and_chamber_scoped_overlay_population():
    unanimous = {"consensus": 4.0, "raters": [
        {"score": 4.0}, {"score": 4.0}, {"score": 4.0}]}
    split = {"consensus": 3.0, "raters": [{"score": 4.0}, {"score": 2.0}]}
    disagreeing = {"consensus": 0.0, "raters": [{"score": 4.0}, {"score": -4.0}]}
    assert is_unanimously_safe(unanimous) is True
    assert is_unanimously_safe(split) is False
    # Raters split across parties is not "unanimously safe" for anybody.
    assert is_unanimously_safe(disagreeing) is False
    # The House slope is fitted only on seats some rater declines to call
    # safe, so unanimously-safe House seats stay out of the overlay...
    assert overlay_consensus(unanimous, "house") is None
    assert overlay_consensus(split, "house") == 3.0
    # ...while the Senate pages list every seat, safe ones included, so the
    # Senate slope is fitted across the full range and nothing is excluded.
    assert overlay_consensus(unanimous, "senate") == 4.0
    assert overlay_consensus(None, "senate") is None


class _StubPrediction:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.model = "core"
        self.calibration = None
        self.calibration_weight = 0.25


def _overlay(weights, sigma=None):
    overlay = RatingOverlay("house")
    overlay.slope = 4.0
    overlay.weights = weights
    overlay.sigma = sigma or {}
    overlay.fit_meta = {"fitted": True}
    return overlay


def test_overlay_shifts_the_margin_toward_the_consensus():
    overlay = _overlay({"unpolled": 0.5}, {"unpolled": 6.0})
    context = overlay.cycle_context([0.0, 2.0, -2.0], [0.0, 1.0, -1.0])
    blended, detail = overlay.apply(_StubPrediction(0.0, 20.0), 2.0, context,
                                    polled=False)
    # implied = level 0 + slope 4 * (2 - mean consensus 0) = +8; half weight.
    assert detail["applied"] is True
    assert blended.mean == pytest.approx(4.0)
    assert detail["ratings_implied_margin"] == pytest.approx(8.0)
    assert detail["margin_shift"] == pytest.approx(4.0)
    # Fitted rated-seat sigma replaces the model's pooled one, inflated for the
    # vintage gap but never widened past the model's own value.
    assert blended.sigma == pytest.approx(6.0 * 1.45)
    assert blended.sigma < 20.0


def test_overlay_is_inert_without_a_rating_or_with_a_zero_weight():
    overlay = _overlay({"unpolled": 0.5}, {"unpolled": 6.0})
    context = overlay.cycle_context([0.0], [0.0])
    base = _StubPrediction(3.0, 20.0)
    unrated, detail = overlay.apply(base, None, context, polled=False)
    assert unrated is base and detail["applied"] is False

    # A stratum the held-out fit gave zero weight must be left completely
    # alone -- including its sigma, which is fitted on a blend that is not
    # happening and would otherwise be substituted anyway.
    zero = _overlay({"polled": 0.0}, {"polled": 32.0})
    same, detail = zero.apply(base, 2.0, zero.cycle_context([0.0], [0.0]),
                              polled=True)
    assert same is base
    assert detail["applied"] is False
    assert same.sigma == 20.0


def test_overlay_declines_to_fit_without_enough_rated_history():
    assert RatingOverlay("house").fit([]).is_fitted is False


def test_states_that_enacted_new_2026_maps_are_all_registered():
    """Every state with a new map for 2026 must be flagged, or its districts
    keep a prior margin describing boundaries that no longer exist.

    Tennessee and Alabama were missing, which is how TN-09 came to publish as a
    toss-up off Steve Cohen's pre-split D+48 Memphis prior while all ten
    handicappers rated the seat Republican.
    """
    enacted = {"AL", "CA", "FL", "LA", "MO", "NC", "OH", "TN", "TX", "UT"}
    for state in enacted:
        assert current_map_cycle(state) == 2026, state
        # A 2024 result predates the 2026 map, so it must read as stale.
        assert prior_is_stale(state, 2024, 2026) is True, state
        assert state in NET_DEM_SEAT_SHIFT, state
    # A state that left its districts in place keeps the post-census baseline.
    for state in ("AR", "IN", "KS", "MD", "NY", "SC"):
        assert current_map_cycle(state) == 2022, state
        assert prior_is_stale(state, 2024, 2026) is False, state


class _Row:
    def __init__(self, chamber="house", poll_count=0, redrawn=False, consensus=None):
        self.chamber = chamber
        self.poll_count = poll_count
        summary = None
        if consensus is not None:
            n = 6
            summary = {"consensus": consensus, "n_raters": n,
                       "raters": [{"score": consensus}] * n}
        self.detail = {"redrawn": redrawn, "ratings": summary}


def test_redrawn_seats_are_their_own_overlay_stratum():
    from app.ratings import _stratum
    assert _stratum(_Row(poll_count=0)) == "unpolled"
    assert _stratum(_Row(poll_count=3)) == "polled"
    # Redrawn wins over both: the prior is known-stale, which is the stronger
    # statement about how much the model deserves to be trusted here.
    assert _stratum(_Row(poll_count=3, redrawn=True)) == "redrawn"


def test_unanimous_safe_exclusion_is_lifted_for_redrawn_seats():
    unanimous = {"consensus": -4.0, "n_raters": 10,
                 "raters": [{"score": -4.0}] * 10}
    # A settled House seat keeps the model's own prediction...
    assert overlay_consensus(unanimous, "house") is None
    assert overlay_consensus(unanimous, "house", redrawn=False) is None
    # ...but on a redrawn seat the stale prior is the thing that is wrong, and
    # a unanimous rating is the only current evidence about the new district.
    assert overlay_consensus(unanimous, "house", redrawn=True) == -4.0
    # The Senate lists every seat, so nothing is ever excluded there.
    assert overlay_consensus(unanimous, "senate") == -4.0


def test_overlay_weight_grid_reaches_the_redrawn_ceiling():
    from app.ratings import (WEIGHT_GRID, MAX_OVERLAY_WEIGHT,
                             MAX_REDRAWN_OVERLAY_WEIGHT)
    # The grid has to span the redrawn ceiling or that stratum silently caps
    # at the ordinary weight and the raised ceiling does nothing.
    assert max(WEIGHT_GRID) >= MAX_REDRAWN_OVERLAY_WEIGHT
    assert MAX_REDRAWN_OVERLAY_WEIGHT > MAX_OVERLAY_WEIGHT
    assert MAX_OVERLAY_WEIGHT in WEIGHT_GRID


def test_overlay_uses_the_redrawn_weight_when_applying():
    overlay = _overlay({"unpolled": 0.5, "redrawn": 1.0},
                       {"unpolled": 6.0, "redrawn": 5.0})
    context = overlay.cycle_context([0.0, 4.0, -4.0], [0.0, 1.0, -1.0])
    # A stale-prior seat the model puts at D+22 with the raters at Safe R.
    blended, detail = overlay.apply(_StubPrediction(22.0, 20.0), -4.0, context,
                                    polled=False, redrawn=True)
    assert detail["stratum"] == "redrawn"
    assert detail["blend_weight"] == 1.0
    # At full weight the model's stale margin is replaced outright.
    assert blended.mean == pytest.approx(-16.0)
    assert detail["ratings_implied_margin"] == pytest.approx(-16.0)
