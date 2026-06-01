# Derived Reference-Path Artifacts

This directory contains reviewed, derived reference-path artifacts used by the
release. It does not contain raw trajectory files, raw scene caches, map images,
or model-training tensors.

## Contents

- `reference_paths/candidates/`: cluster-average candidate path seeds.
- `reference_paths/manual_selection/`: reviewed candidate decisions with
  accepted, deleted, and rejected cluster IDs.
- `reference_paths/paths/`: final spline-fitted reference paths.
- `reference_paths/rejects/`: `ignore_track` prototypes generated from rejected
  candidate groups.
- `reference_paths/qc/`: JSON QC reports. Visual overlay PNGs are excluded.
- `reference_paths/inter_path_geom/`: compact NPZ geometry caches for path
  interaction features.
- `assignments/`: final per-scene raw-track assignment/filter JSON files.
- `assignments_qc/`: assignment summary reports.

## Scene Counts

| Dataset | Paths | Manual selections | Assignments |
| --- | ---: | ---: | ---: |
| CitySim | 4 | 4 | 4 |
| DJI | 68 | 68 | 68 |
| HighD | 60 | 60 | 60 |
| NGSIM | 6 | 6 | 6 |
| inD | 33 | 33 | 33 |
| sinD | 56 | 56 | 56 |

The raw `SceneRaw` pickle cache is intentionally excluded because it is a large
raw-data-derived cache.
