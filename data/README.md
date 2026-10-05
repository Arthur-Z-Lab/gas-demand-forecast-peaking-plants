# Data

`HDQS.xlsx` contains **representative samples of the primary dataset** used in the paper:
the first 500 and the last 500 daily natural gas consumption observations of the primary gas-fired
peaking plant, with the middle section withheld because the complete operational dataset cannot be
released for confidentiality reasons.

Layout: a single column `HDQS` (daily load, m3/day), 1000 rows. The two segments are concatenated,
so the file is discontinuous in time. It is intended for running and inspecting the training pipeline
(`python run.py --data data/HDQS.xlsx --col HDQS ...`), not for reproducing the exact numbers reported
in the paper.

To use your own series, pass `--data <file> --col <column>`; both Excel and CSV with one numeric
column of consecutive daily values are accepted.
