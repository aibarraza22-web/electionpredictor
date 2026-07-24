import textwrap

from app.ingest import csv_results, fte_polls, legislators, medsl

RAW_POLLS_HEADER = (
    "poll_id,question_id,race_id,cycle,location,type_simple,race,pollster,"
    "pollster_rating_id,aapor_roper,inactive,methodology,transparency_score,"
    "partisan,polldate,electiondate,time_to_election,samplesize,cand1_name,"
    "cand1_id,cand1_party,cand1_pct,cand1_actual,cand2_name,cand2_id,"
    "cand2_party,cand2_pct,cand2_actual,margin_poll,margin_actual")


def _poll_line(question_id, race_id, cycle, location, kind, electiondate,
               cand1_party="DEM", cand1_pct=48, cand1_actual=49.0,
               cand2_party="REP", cand2_pct=46, cand2_actual=51.0):
    return (f"{question_id},{question_id},{race_id},{cycle},{location},{kind},"
            f"{cycle}_{kind}_{location},Test Poll Co,1,FALSE,FALSE,Live Phone,8,NA,"
            f"{cycle}-10-01,{electiondate},30,600,Alice,1,{cand1_party},{cand1_pct},"
            f"{cand1_actual},Bob,2,{cand2_party},{cand2_pct},{cand2_actual},"
            f"{cand1_pct - cand2_pct},{cand1_actual - cand2_actual}")


def test_fte_parse_orientation_and_results():
    csv_text = "\n".join([
        RAW_POLLS_HEADER,
        _poll_line(1, 100, 2022, "NC", "Sen-G", "2022-11-08"),
        # REP listed first: margins must be re-oriented to dem-rep
        _poll_line(2, 100, 2022, "NC", "Sen-G", "2022-11-08",
                   cand1_party="REP", cand1_pct=50, cand1_actual=51.0,
                   cand2_party="DEM", cand2_pct=44, cand2_actual=49.0),
        # not a two-party D-R race: skipped
        _poll_line(3, 101, 2022, "CA", "Sen-G", "2022-11-08",
                   cand2_party="DEM", cand2_pct=30, cand2_actual=39.0),
        # house race with district
        _poll_line(4, 102, 2022, "OH-01", "House-G", "2022-11-08"),
    ])
    polls, results = fte_polls.parse(csv_text.encode())
    assert len(polls) == 3 and len(results) == 2
    reoriented = next(p for p in polls if p["external_id"] == "2")
    assert reoriented["dem_margin"] == -6
    house = next(r for r in results if r["chamber"] == "house")
    assert house["seat_key"] == "house-OH-01" and house["dem_margin"] == -2.0


def test_fte_parse_runoff_keeps_decisive_result():
    csv_text = "\n".join([
        RAW_POLLS_HEADER,
        _poll_line(1, 200, 2022, "GA", "Sen-G", "2022-11-08",
                   cand1_actual=49.4, cand2_actual=48.5),
        _poll_line(2, 200, 2022, "GA", "Sen-G", "2022-12-06",
                   cand1_actual=51.4, cand2_actual=48.6),
    ])
    _, results = fte_polls.parse(csv_text.encode())
    assert len(results) == 1
    assert abs(results[0]["dem_margin"] - 2.8) < 1e-9


def test_legislators_parse_builds_2026_universe():
    import json
    members = [
        {"id": {}, "name": {"official_full": "Rep One"},
         "terms": [{"type": "rep", "state": "OH", "district": 1,
                    "party": "Republican", "start": "2025-01-03", "end": "2027-01-03"}]},
        {"id": {}, "name": {"official_full": "Delegate"},
         "terms": [{"type": "rep", "state": "DC", "district": 0,
                    "party": "Democrat", "start": "2025-01-03", "end": "2027-01-03"}]},
        {"id": {}, "name": {"official_full": "Class Two"},
         "terms": [{"type": "sen", "state": "NC", "class": 2,
                    "party": "Republican", "start": "2021-01-03", "end": "2027-01-03"}]},
        {"id": {}, "name": {"official_full": "Appointed Three"},
         "terms": [{"type": "sen", "state": "OH", "class": 3,
                    "party": "Republican", "start": "2025-01-21", "end": "2027-01-03"}]},
        {"id": {}, "name": {"official_full": "Not Up"},
         "terms": [{"type": "sen", "state": "VT", "class": 1,
                    "party": "Independent", "start": "2025-01-03", "end": "2031-01-03"}]},
    ]
    rows, dem_not_up = legislators.parse(json.dumps(members).encode())
    keys = {r["seat_key"]: r for r in rows}
    assert set(keys) == {"house-OH-01", "senate-NC", "senate-OH-special"}
    assert keys["senate-OH-special"]["party"] == "R"
    assert dem_not_up == 1  # the Independent counts toward the Democratic caucus


def test_bundled_senate_file_is_real_and_parses():
    # Guards the root-cause Senate fix: the bundled Senate returns must exist,
    # parse, and match known certified margins (so the model never silently
    # falls back to synthetic Senate data again).
    assert medsl.BUNDLED_SENATE_FILE.exists(), "bundled Senate returns missing"
    rows = medsl.parse(medsl.BUNDLED_SENATE_FILE.read_bytes(), "senate")
    cycles = sorted({r["cycle"] for r in rows})
    assert cycles[0] <= 2004 and 2024 in cycles and len(cycles) >= 8
    by = {(r["cycle"], r["state"]): r["dem_margin"] for r in rows}
    # spot checks vs reality (two-party Dem margin)
    assert abs(by[(2018, "TX")] - (-2.6)) < 1.5   # Cruz beat O'Rourke ~2.6
    assert abs(by[(2012, "MA")] - (7.6)) < 1.5    # Warren ~+7.6
    assert by[(2014, "NC")] < 0                    # Tillis (R) won


def test_csv_results_importer(tmp_path, temp_db):
    from app import store
    path = tmp_path / "official.csv"
    path.write_text(textwrap.dedent("""\
        cycle,chamber,state,district,dem_votes,rep_votes
        2024,house,OH,1,180000,170000
        2024,senate,OH,,2000000,2200000
        2024,house,ZZ,1,1,1
    """))
    summary = csv_results.ingest_file(path)
    assert summary["results"] == 2
    rows = store.all_results()
    house = next(r for r in rows if r["chamber"] == "house")
    assert house["seat_key"] == "house-OH-01" and house["winner_party"] == "D"
