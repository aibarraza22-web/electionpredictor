"""Poll-free track-record model (research claim T-004) and its inputs."""
from random import Random

import pytest

from app import redistricting, track_record as tr
from app.ingest import ballot_2026, cook_pvi


# --- Cook PVI parsing: the article changed format three times --------------

PVI_2010_STYLE = """==List of PVIs==
===By congressional district===
{| class="wikitable sortable"
|-
| Alaska
| [[Alaska's At-large congressional district|At-large]]
| {{party shading/Republican}} | <span style="display:none">086</span>R+13
|-
| California
| [[California's 28th congressional district|28th]]
| {{party shading/Democratic}} | <span style="display:none">123</span>D+20
|}
===By state===
{| class="wikitable sortable"
|-
| [[Alabama]]
| {{party shading/Republican}} | <span style="display:none">086</span>R+13
|}
"""
PVI_2018_STYLE = """==By congressional district==
|-
| {{ushr|California|26|X}}
| {{party shading/Democratic}} | {{sort|092|D+7}}
|-
| {{ushr|Ohio|9|X}}
| EVEN
"""
PVI_2026_STYLE = """==By congressional district==
|-
| {{ushr|Texas|9|X}}
| {{Shading PVI|R|9}}
|-
| {{ushr|Texas|34|X}}
| {{Shading PVI|R|value=3}}
|-
| {{ushr|Wyoming|AL|X}}
| {{Shading PVI|R|23}}
|}
==By state==
{| class="wikitable"
|-
| [[West Virginia]]
| {{Shading PVI|R|21}}
|-
| [[Wyoming]]
| {{Shading PVI|R|23}}
|}
==See also==
[[Wyoming's at-large congressional district]]
"""


def test_pvi_parser_handles_every_article_format():
    old = cook_pvi.parse_districts(PVI_2010_STYLE)
    assert old == {("AK", 1): -13.0, ("CA", 28): 20.0}
    assert cook_pvi.parse_states(PVI_2010_STYLE) == {"AL": -13.0}
    mid = cook_pvi.parse_districts(PVI_2018_STYLE)
    assert mid == {("CA", 26): 7.0, ("OH", 9): 0.0}
    new = cook_pvi.parse_districts(PVI_2026_STYLE)
    assert new == {("TX", 9): -9.0, ("TX", 34): -3.0, ("WY", 1): -23.0}
    # "West Virginia" must not be read as Virginia; the table's last row counts
    assert cook_pvi.parse_states(PVI_2026_STYLE) == {"WV": -21.0, "WY": -23.0}


def test_bundled_pvi_vintages_cover_2026_on_the_new_maps():
    districts, states = cook_pvi.load_vintages()
    assert sum(1 for (c, _s, _d) in districts if c == 2026) == 435
    assert sum(1 for (c, _s) in states if c == 2026) == 50
    # redrawn for 2026: the Memphis 9th was split, Texas' 9th became R-leaning
    assert districts[(2026, "TN", 9)] < 0 < districts[(2024, "TN", 9)]
    assert districts[(2026, "TX", 9)] < 0 < districts[(2024, "TX", 9)]


# --- 2026 ballot status ------------------------------------------------------

HOUSE_ROWS = """
|-
!{{ushr|NE|2|X}}
|{{shading PVI|D|3}}
|{{sortname|Don|Bacon}}
|style="background-color:#FFB6B6" |Republican
|[[2016 United States House of Representatives elections in Nebraska|2016]]
|style="background:#DDDDDD" |Incumbent retiring<ref name="BaconNE"/>
|nowrap |{{plainlist}}
*{{Party stripe|Democratic Party (US)}}Denise Powell (Democratic)
{{endplainlist}}
|-
!rowspan=2 |{{ushr|TX|33|X}}
|rowspan=2 {{shading PVI|D|18}}
|{{sortname|Jasmine|Crockett}}
|style="color:black;background-color:#B0CEFF" |Democratic
|[[2022 United States House of Representatives elections in Texas|2022]]
|Incumbent renominated
|rowspan=2 |{{plainlist}}
*{{Party stripe|Republican Party (US)}}Patrick Gillespie (Republican)
{{endplainlist}}
"""
SENATE_ROWS = """
|-
! [[2026 United States Senate special election in Ohio|Ohio]]<br />(Class 3)
| {{Shading PVI|R|5}}
| [[Jon Husted]]
| {{Party shading/Republican}} | Republican
| 2025 {{small|(appointed)}}
| data-sort-value=0 | Interim appointee nominated
| nowrap | {{Plainlist |
*{{Party stripe|Democratic Party (United States)}}[[Sherrod Brown]] (Democratic)
}}
|-
! [[2026 United States Senate election in Maine|Maine]]
| {{Shading PVI|D|4}}
| {{sortname|Susan|Collins}}
| {{Party shading/Republican}} | Republican
| data-sort-value=1 | Incumbent renominated
| nowrap | {{Plainlist |
*{{Party stripe|Democratic Party (US)}}Someone (Democratic)
}}
"""


def test_ballot_status_parser():
    house = {r["seat_key"]: r for r in ballot_2026.parse_house(HOUSE_ROWS)}
    assert house["house-NE-02"]["incumbent_running"] == 0
    assert house["house-NE-02"]["incumbent_party"] == "R"
    # rowspan rows parse too; the candidate list's parties never leak in
    assert house["house-TX-33"]["incumbent_running"] == 1
    assert house["house-TX-33"]["incumbent_party"] == "D"
    senate = {r["seat_key"]: r for r in ballot_2026.parse_senate(SENATE_ROWS)}
    # appointed senators are not "incumbents" in the fitted sense (see RUNNING)
    assert senate["senate-OH-special"]["incumbent_running"] == 0
    assert senate["senate-ME"]["incumbent_running"] == 1


# --- incumbency and redistricting ---------------------------------------------

def test_incumbency_flags_follow_prior_winners():
    from app.incumbency import IncumbencyLookup, surname
    assert surname("ROBERT C. \"BOBBY\" SCOTT JR.") == "SCOTT"
    lookup = IncumbencyLookup.__new__(IncumbencyLookup)
    lookup._flags = {}
    lookup._derive({
        (2020, "house-XX-01"): {"D": ("SMITH", 60.0), "R": ("JONES", 40.0)},
        (2022, "house-XX-01"): {"D": ("SMITH", 55.0), "R": ("BROWN", 45.0)},
        # Smith moved districts after a redraw: still an incumbent
        (2022, "house-XX-02"): {"D": ("SMITH", 30.0), "R": ("GREEN", 70.0)},
        (2022, "house-XX-03"): {"D": ("NEW", 30.0), "R": ("OTHER", 70.0)},
    }, "house")
    assert lookup.value(2022, "house-XX-01") == 1.0
    assert lookup.value(2022, "house-XX-02") == 1.0
    assert lookup.value(2022, "house-XX-03") == 0.0
    assert lookup.value(2026, "house-ZZ-01") is None


def test_map_changes_cover_census_and_mid_decade_redraws():
    assert redistricting.map_changed("PA", 2016, 2018)       # court map
    assert not redistricting.map_changed("PA", 2018, 2020)
    assert redistricting.map_changed("WI", 2020, 2022)       # census
    assert not redistricting.map_changed("WI", 2022, 2024)
    assert redistricting.map_changed("NY", 2022, 2024)       # 2024 redraw
    assert redistricting.map_changed("OH", 2024, 2026)       # 2026 redraw
    assert not redistricting.map_changed("OH", None, 2026)


# --- the model -----------------------------------------------------------------

def _history(seed=3, cycles=range(2008, 2026, 2), n=240):
    """Synthetic House history where PVI and incumbency drive outcomes."""
    rng = Random(seed)
    lean = [rng.gauss(0, 12) for _ in range(n)]
    history = {}
    for cycle in cycles:
        env, mid = tr.environment(cycle)
        national = rng.gauss(0, 3)
        rows = []
        for i, pvi in enumerate(lean):
            inc = rng.choice([1.0, -1.0, 0.0])
            y = 2.0 * pvi + 4.0 * inc + national + rng.gauss(0, 5)
            rows.append(tr.SeatRow(seat_key=f"house-ZZ-{i:02d}", chamber="house",
                                   state=("AA", "BB", "CC")[i % 3], district=str(i + 1),
                                   cycle=cycle, y=y, prior=None, pvi=pvi, lean=0.0,
                                   inc=inc, env=env, mid=mid))
        history[cycle] = rows
    return history


def test_walk_forward_is_out_of_sample_and_picks_the_right_system():
    history = _history()
    records = tr.walk_forward(history)
    # the first cycle has no training data and is never scored
    assert min(c for c, _ in records) > min(history)
    # changing a LATER cycle's outcomes cannot change an earlier prediction
    altered = {c: rows for c, rows in history.items()}
    altered[2024] = [tr.SeatRow(**{**r.__dict__, "y": -r.y}) for r in history[2024]]
    again = tr.walk_forward(altered)
    key = next(k for k in records if k[0] == 2020)
    assert again[key].preds == records[key].preds
    ranking = tr.rank_systems(list(records.values()))
    assert ranking[0] in {"pvi_inc", "full", "ensemble"}


def test_systems_train_only_where_their_input_exists():
    history = _history()
    rows = [r for rs in history.values() for r in rs]
    # add PVI-less rows whose outcomes would badly distort a PVI fit
    noise = [tr.SeatRow(**{**r.__dict__, "pvi": None, "y": 40.0 * r.inc}) for r in rows[:300]]
    clean = tr.fit_systems(rows)["pvi_inc"]
    mixed = tr.fit_systems(rows + noise)["pvi_inc"]
    assert clean.weights == pytest.approx(mixed.weights)
    # and a system missing its defining input is never used for that race
    row = tr.SeatRow(seat_key="house-ZZ-99", chamber="house", state="AA", district="99",
                     cycle=2026, y=None, pvi=None)
    assert tr.pick_for(["pvi_inc", "pvi", "ensemble"], row) == "ensemble"


def test_local_uncertainty_tracks_each_seats_own_record():
    records = []
    for cycle in (2018, 2020, 2022, 2024):
        for i in range(60):
            miss = 15.0 if i == 0 else 1.0          # seat 0 is always badly missed
            sign = 1 if (cycle // 2) % 2 else -1
            records.append(tr.Scored(cycle, f"house-ZZ-{i:02d}", "AA", 5.0 + sign * miss,
                                     {c: 5.0 for c in tr.CANDIDATES}, pvi=1.0, prior=1.0))
    unc = tr.Uncertainty(records, "pvi_inc", "house")
    noisy = unc.local(2026, "house-ZZ-00", "AA")
    steady = unc.local(2026, "house-ZZ-05", "AA")
    assert noisy["sigma"] > steady["sigma"] >= tr.MIN_SIGMA
    assert noisy["national"] == steady["national"]  # the shared part is shared


def test_calibration_is_slope_only():
    rng = Random(5)
    rows = []
    for _ in range(400):
        z = rng.gauss(0, 1.5)
        rows.append((z * 10.0, 10.0, 1.0 if rng.random() < 1 / (1 + 2.718 ** (-3 * z)) else -1.0))
    a, b = tr.fit_calibration(rows)
    assert a == 0.0 and b > 1.6          # sharper than the raw normal reading
    assert tr.fit_calibration(rows[:10]) is None
    assert tr.probability(0.0, 10.0, (a, b)) == pytest.approx(0.5)


def test_poll_free_gate():
    from app import gates
    ok = {"races": 470, "with_pvi": 470, "with_ballot_status": 470}
    assert gates.check_poll_free_inputs(ok)["passed"]
    with pytest.raises(gates.GateFailure):
        gates.check_poll_free_inputs({**ok, "with_pvi": 400})
    with pytest.raises(gates.GateFailure):
        gates.check_poll_free_inputs({**ok, "with_ballot_status": 100})


# --- results data: independents, fusion tickets, mislabelled rows -------------------

def _tally(year, state, rows):
    from app.ingest import party_labels
    prepared = [{"candidate": name, "votes": votes, "labels": labels}
                for name, votes, labels in rows]
    return party_labels.tally(year, state, prepared)


def test_independent_who_caucuses_with_democrats_is_on_the_democratic_side():
    """Maine 2018: Angus King (I) 54.3%, Republican 35.3%, Democrat 10.4%. The
    raw party column made this 'R+54'; King's side is the Democratic one."""
    t = _tally(2018, "ME", [
        ("ANGUS S. KING, JR.", 344575, ("INDEPENDENT", "OTHER")),
        ("ERIC L. BRAKEY", 223502, ("REPUBLICAN", "REPUBLICAN")),
        ("ZAK RINGELSTEIN", 66268, ("DEMOCRAT", "DEMOCRAT")),
        ("OTHERS", 64, ("", "OTHER"))])
    assert t["top_side"] == "D" and t["D"] == pytest.approx(344575 + 66268)
    margin = (t["D"] - t["R"]) / (t["D"] + t["R"]) * 100
    assert 28 < margin < 31  # D+29, not R+54


def test_fusion_ticket_votes_are_summed_per_candidate():
    t = _tally(1992, "CT", [
        ("CHRISTOPHER J. DODD", 600000, ("DEMOCRAT",)),
        ("CHRISTOPHER J. DODD", 250000, ("A CONNECTICUT PARTY", "OTHER")),
        ("BROOK JOHNSON", 450000, ("REPUBLICAN",))])
    assert t["D"] == 850000 and t["R"] == 450000


def test_blank_and_mislabelled_party_rows_are_resolved():
    # Wyoming 2020: no party for anyone in the source file
    t = _tally(2020, "WY", [("CYNTHIA M. LUMMIS", 198100, ("", "OTHER")),
                            ("MERAV BEN DAVID", 72766, ("", "OTHER")),
                            ("OVER VOTES", 165, ("", "OTHER"))])
    assert t["top_side"] == "R" and t["D"] == 72766
    # 'DEMOCRATIC' simplified to OTHER (Duckworth 2022) and Minnesota's
    # Independent-Republican label are both read from the detailed label
    from app.ingest import party_labels
    assert party_labels.resolve(2022, "IL", "TAMMY DUCKWORTH", "DEMOCRATIC", "OTHER") == "D"
    assert party_labels.resolve(1992, "MN", "JIM RAMSTAD", "INDEPENDENT-REPUBLICAN") == "R"
    # an ordinary independent or third party stays 'other'
    assert party_labels.resolve(2018, "NM", "GARY E JOHNSON", "LIBERTARIAN") is None
    assert party_labels.resolve(2014, "SD", "LARRY PRESSLER", "INDEPENDENT") is None


def test_bundled_results_match_certified_seat_totals():
    from app.ingest import medsl
    house = medsl.parse(medsl.BUNDLED_HOUSE_FILE.read_bytes(), "house")
    certified = {2010: 193, 2012: 201, 2014: 188, 2018: 235, 2020: 222, 2022: 213, 2024: 215}
    for year, democrats in certified.items():
        assert sum(1 for r in house if r["cycle"] == year and r["winner_party"] == "D") == democrats
    senate = {(r["cycle"], r["seat_key"]): r for r in
              medsl.parse(medsl.BUNDLED_SENATE_FILE.read_bytes(), "senate")}
    for cycle in (2012, 2018, 2024):  # King won Maine as an independent each time
        assert senate[(cycle, "senate-ME")]["winner_party"] == "D"
        assert senate[(cycle, "senate-ME")]["dem_margin"] > 20
    assert senate[(2020, "senate-ME")]["dem_margin"] < 0     # Collins


def test_race_history_names_the_independent_who_won():
    from app.ingest import party_labels
    assert "King" in party_labels.aligned_note(2024, "ME")
    assert party_labels.aligned_note(2020, "ME") is None


def test_replace_results_overwrites_stale_rows(temp_db):
    """A corrected parse must replace stored rows; insert_rows ignores duplicates."""
    from app import store
    base = {"cycle": 2018, "chamber": "senate", "state": "ME", "district": None,
            "seat_key": "senate-ME", "dem_pct": 1.0, "rep_pct": 99.0,
            "winner_party": "R", "source": "medsl-constituency-returns"}
    store.replace_results("medsl-constituency-returns", "senate", [{**base, "dem_margin": -54.3}])
    store.replace_results("medsl-constituency-returns", "senate",
                          [{**base, "dem_margin": 29.5, "winner_party": "D"}])
    rows = store.all_results("senate")
    assert len(rows) == 1 and rows[0]["dem_margin"] == 29.5 and rows[0]["winner_party"] == "D"


# --- personal vote and the poll layer ---------------------------------------------------

def _scored(cycle, seat, y, ref_pred, state="AA", pvi=1.0, inc=1.0):
    return tr.Scored(cycle, seat, state, y, {c: ref_pred for c in tr.CANDIDATES},
                     pvi=pvi, inc=inc, chamber="house", raw={tr.PERSONAL_VOTE_REFERENCE: ref_pred})


def test_personal_vote_only_when_the_same_incumbent_runs_again_on_the_same_map():
    records = {(2022, "house-AA-01"): _scored(2022, "house-AA-01", y=10.0, ref_pred=2.0)}
    row = tr.SeatRow(seat_key="house-AA-01", chamber="house", state="AA", district="1",
                     cycle=2024, y=None, pvi=1.0, inc=1.0)
    assert tr.personal_vote_term("house", row, records) == pytest.approx(0.3 * 8.0)
    # open seat: nothing to carry forward
    assert tr.personal_vote_term("house", tr.SeatRow(**{**row.__dict__, "inc": 0.0}), records) == 0.0
    # the incumbent is on the OTHER side of the party that won last time
    assert tr.personal_vote_term("house", tr.SeatRow(**{**row.__dict__, "inc": -1.0}), records) == 0.0
    # districts redrawn between the two elections (Ohio, 2024 -> 2026)
    redrawn = tr.SeatRow(**{**row.__dict__, "state": "OH", "cycle": 2026})
    records[(2024, "house-AA-01")] = _scored(2024, "house-AA-01", 10.0, 2.0, state="OH")
    assert tr.personal_vote_term("house", redrawn, records) == 0.0
    # the Senate weight is zero: tested, mixed, not adopted
    assert tr.personal_vote_term("senate", row, records) == 0.0


def test_poll_layer_is_weighted_by_how_much_polling_there_is():
    one = tr.SeatRow(seat_key="s", chamber="senate", state="AA", district=None, cycle=2026,
                     y=None, pvi=0.0, inc=0.0, poll=10.0, poll_n=1)
    ten = tr.SeatRow(**{**one.__dict__, "poll_n": 10})
    none = tr.SeatRow(**{**one.__dict__, "poll": None, "poll_n": 0})
    feature = lambda r: tr.vector(r, ["poll"])[1][-1]  # noqa: E731
    assert feature(one) == pytest.approx(10.0 / 3.0)        # a lone poll counts a third
    assert feature(ten) == pytest.approx(10.0 * 10 / 12)    # ten polls count 83%
    assert feature(none) == 0.0
    assert tr.usable(tr.POLL_LAYER, one) and not tr.usable(tr.POLL_LAYER, none)


def test_polled_races_use_the_polled_ranking_and_others_do_not():
    polled = tr.SeatRow(seat_key="s", chamber="house", state="AA", district="1", cycle=2026,
                        y=None, pvi=1.0, poll=3.0, poll_n=4)
    unpolled = tr.SeatRow(**{**polled.__dict__, "poll": None, "poll_n": 0})
    ranking, polled_ranking = ["pvi_inc", "ensemble"], [tr.POLL_LAYER, "pvi_inc"]
    assert tr.pick_for(ranking, polled, polled_ranking) == tr.POLL_LAYER
    assert tr.pick_for(ranking, unpolled, polled_ranking) == "pvi_inc"
    assert tr.pick_for(ranking, polled, []) == "pvi_inc"   # too little polled history to judge


def test_multi_candidate_polls_are_not_treated_as_head_to_heads():
    """Two Alaska polls read D 48.4 / R 2.7 (name-matching found a minor
    Republican) and made Alaska 'Safe Democratic'."""
    base = {"cycle": 2026, "seat_key": "senate-AK", "poll_date": "2026-06-01"}
    rows = [{**base, "dem_pct": 48.4, "rep_pct": 2.7, "dem_margin": 45.7},      # AK: dropped
            {**base, "dem_pct": 22.0, "rep_pct": 45.0, "dem_margin": -23.0},    # independent splits MT
            {**base, "dem_pct": 44.0, "rep_pct": 46.0, "dem_margin": -2.0}]     # a real head-to-head
    lookup = tr.poll_lookup(rows)
    average, n, _ = lookup.average(2026, "senate-AK", "2026-09-01")
    assert n == 1 and average == pytest.approx(-2.0)


def test_votehub_rejects_multi_candidate_polls():
    from datetime import date, timedelta
    from app.ingest import votehub
    fresh = (date.today() - timedelta(days=5)).isoformat()
    index = {"senate-AK": [("sullivan", "REP"), ("peltola", "DEM"), ("miller", "REP")]}
    poll = {"poll_type": "us-senator", "subject": "2026 Alaska", "end_date": fresh, "id": "a",
            "sample_size": 800, "population": "lv",
            "answers": [{"choice": "Mary Peltola", "pct": 48.4}, {"choice": "Miller", "pct": 2.7}]}
    assert votehub._normalize(poll, index) is None


# --- races that are not a Democrat-vs-Republican contest (Alaska) -------------------------

def test_same_party_finals_and_vote_split_races_are_not_partisan_margins():
    """Alaska 2022 (Murkowski vs Tshibaka, both Republicans, Democrat at 11%) read
    'R+78'; Alaska 2016 (a Libertarian at 29%, an independent at 13%) read 'R+58'."""
    ak22 = _tally(2022, "AK", [("MURKOWSKI, LISA", 114118, ("",)), ("TSHIBAKA, KELLY C.", 112101, ("",)),
                               ("CHESBRO, PATRICIA R.", 28233, ("",)), ("KELLEY, BUZZ A.", 8575, ("",))])
    assert ak22["top_side"] == "R" and ak22["comparable"] is False
    ak16 = _tally(2016, "AK", [("LISA MURKOWSKI", 138149, ("REPUBLICAN",)), ("JOE MILLER", 90825, ("LIBERTARIAN",)),
                               ("MARGARET STOCK", 41194, ("INDEPENDENT",)), ("RAY METCALFE", 36200, ("DEMOCRAT",))])
    assert ak16["comparable"] is False
    # ordinary races are untouched: a normal two-way race, and Maine 2018 with King
    assert _tally(2020, "AK", [("DAN SULLIVAN", 191112, ("REPUBLICAN",)), ("AL GROSS", 146068, ("DEMOCRAT",)),
                               ("JOHN WAYNE HOWE", 16597, ("ALASKAN INDEPENDENCE",))])["comparable"] is True
    assert _tally(2018, "ME", [("ANGUS S. KING, JR.", 344575, ("INDEPENDENT",)), ("ERIC L. BRAKEY", 223502, ("REPUBLICAN",)),
                               ("ZAK RINGELSTEIN", 66268, ("DEMOCRAT",))])["comparable"] is True


def test_non_comparable_races_are_stored_like_uncontested_ones_with_the_real_winner():
    from app.ingest import medsl
    senate = {(r["cycle"], r["seat_key"]): r for r in
              medsl.parse(medsl.BUNDLED_SENATE_FILE.read_bytes(), "senate")}
    for cycle in (2010, 2016, 2022):
        row = senate[(cycle, "senate-AK")]
        assert row["dem_margin"] == -100.0 and row["winner_party"] == "R"   # excluded from learning
    assert -20 < senate[(2020, "senate-AK")]["dem_margin"] < -5             # a real race stays real
    assert senate[(2014, "senate-AK")]["dem_margin"] > -5


def test_no_best_system_is_crowned_on_a_handful_of_races():
    few = [tr.Scored(2010, f"senate-S{i}", "AA", 3.0, {c: 1.0 for c in tr.CANDIDATES}) for i in range(10)]
    assert tr.rank_systems(few)[0] == tr.ENSEMBLE
    # with enough close races the record decides: make one system right and the rest wrong
    many = []
    for i in range(80):
        preds = {c: -2.0 for c in tr.CANDIDATES}
        preds["pvi_inc"] = 2.0
        many.append(tr.Scored(2012, f"house-AA-{i:02d}", "AA", 3.0, preds))
    assert tr.rank_systems(many)[0] == "pvi_inc"


def test_conflicting_evidence_widens_uncertainty():
    """Maine 2020: polls said D+4.5, every poll-free system said Collins. When the
    systems disagree, or the polls pull the poll layer away from them, sigma
    grows; when they agree it does not."""
    agree = {"persistence": 5.0, "pvi": 5.2, "pvi_inc": 4.8, "core": 5.1, "full": 4.9,
             tr.ENSEMBLE: 5.0, tr.POLL_LAYER: 5.3}
    split = {"persistence": -21.7, "pvi": 13.1, "pvi_inc": -1.2, "core": -13.5, "full": -7.3,
             tr.ENSEMBLE: -6.1, tr.POLL_LAYER: 4.9}
    assert tr.widen(5.0, agree, "pvi_inc") == pytest.approx(5.0, abs=0.1)
    assert tr.widen(5.0, split, "pvi_inc") > 12.0        # the five systems span 35 points (sd 11.7)
    # polled race: widened by the polls-vs-fundamentals gap (11 points), not the spread
    assert tr.disagreement(split, tr.POLL_LAYER) == pytest.approx(11.0)
    assert tr.widen(5.0, split, tr.POLL_LAYER) == pytest.approx((25 + 121) ** 0.5)
    # an unpolled race has no gap term; missing inputs never raise
    assert tr.disagreement({tr.ENSEMBLE: 1.0}, tr.ENSEMBLE) == 0.0
    assert tr.disagreement({}, tr.POLL_LAYER) == 0.0


def test_lean_curve_applies_to_the_house_only():
    house = tr.SeatRow(seat_key="h", chamber="house", state="AA", district="1", cycle=2026, y=None, pvi=-30.0)
    senate = tr.SeatRow(**{**house.__dict__, "chamber": "senate", "seat_key": "s"})
    names, x = tr.vector(house, ["pvi"])
    assert "pvi_curve" in names and x[names.index("pvi_curve")] == pytest.approx(-9.0)
    assert "pvi_curve" not in tr.vector(senate, ["pvi"])[0]


def test_seat_record_reports_the_margin_miss():
    rec = tr.Scored(2020, "senate-ME", "ME", -9.2, {"ensemble": 4.9}, chamber="senate")
    row = tr.SeatRow(seat_key="senate-ME", chamber="senate", state="ME", district=None, cycle=2026, y=None)
    out = tr.seat_record({(2020, "senate-ME"): rec}, row, "ensemble")
    assert out[0]["margin_miss"] == pytest.approx(-14.1) and out[0]["called_correctly"] is False
    blowout = tr.Scored(2020, "senate-ME", "ME", 100.0, {"ensemble": 4.9}, chamber="senate")
    assert tr.seat_record({(2020, "senate-ME"): blowout}, row, "ensemble")[0]["margin_miss"] is None
