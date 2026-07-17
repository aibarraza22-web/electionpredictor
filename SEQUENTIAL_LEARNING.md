# Sequential learning

For each cycle, construct an as-of snapshot using only records available before the vintage; exclude cycle *t* from training; generate and freeze forecasts; score them after results; then append *t* and refit before *t+1*. `walk_forward` asserts both availability and exclusion. Contemporaneous snapshots are immutable; retrospective replays require a distinct model version. Calibration is fit only on earlier frozen predictions.
