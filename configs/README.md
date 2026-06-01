# Configuration Templates

This directory stores publication-safe configuration templates.

- `datasets/ind_demo.yaml` targets the small inD demo/test subset that can be
  included in this repository.
- Other dataset configs are path templates only. Users must provide local paths
  to raw datasets obtained from the original data providers.
- `experiments/demo_ind.yaml` defines the small demo run.

Do not commit local absolute paths, credentials, full processed datasets, or
provider-restricted files.
