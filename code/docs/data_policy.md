# Data Availability

This repository includes demo inputs, reference-path artifacts, preprocessing
scripts, released checkpoints, and metric scripts for reproducing the main code
paths. Full-data experiments use local dataset paths configured under
`code/configs/datasets/`.

## Included

- A small inD demo/test excerpt from recording `04`, limited to 1,929 trajectory
  rows and 13 assigned vehicle tracks.
- Inputs for the Appendix long-tail case figure demo under
  `data/demo/longtail_cases/`.
- Code and scripts for all supported datasets, including INTERACTION.
- Configuration templates that point users to local dataset locations.
- Reference-path processing and filtering code plus the reviewed reference-path
  metadata and track-assignment filters used by this release.
- One compact CN+US-source metatype checkpoint under
  `code/pretrained/metatypes/cn_us_source_metatype/`.
- One inD evaluation checkpoint under `code/pretrained/models/ind_5k_best/`.

## Full-Data Runs

After configuring dataset paths, run the dataset builders under
`code/scripts/datasets/`, then run training, transfer, and evaluation commands from
the repository root. Generated tensors, caches, and paper-scale outputs should
be written under local artifact directories.
