# Configuration Templates

This directory stores publication-safe configuration templates.

- `datasets/ind_demo.yaml` targets the small inD demo/test subset that can be
  included in this repository.
- `datasets/*.yaml` files are path templates for independent dataset builders:
  HighD, inD, CitySim, sinD, NGSIM, DJI, and INTERACTION.
- `reference_paths.yaml` is a path template for candidate mining and assignment
  against local raw datasets.
- `experiments/demo_ind.yaml` defines the small demo run.

Do not commit local absolute paths, credentials, full processed datasets, or
provider-restricted files.
