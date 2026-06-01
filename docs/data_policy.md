# Data Policy

This release intentionally does not include full third-party driving datasets.
The paper uses public or access-controlled trajectory datasets whose providers
retain their own redistribution rules. Publishing full raw or processed copies
inside this repository would create license and attribution risk.

## What Is Included

- A small inD demo/test excerpt from recording `04`, limited to 1,929 trajectory
  rows and 13 assigned vehicle tracks. It is included only for repository smoke
  tests; full inD data remains excluded and governed by the provider terms.
- Code and scripts for all supported datasets, including INTERACTION.
- Configuration templates that point users to local dataset locations.
- Reference-path processing and filtering code plus the reviewed derived
  reference-path metadata and track-assignment filters used by this release.
- One compact DE-source metatype checkpoint under
  `pretrained/metatypes/de_source_metatype/`. It contains trained parameters
  and normalization statistics for the proposed method only; it does not include
  raw or processed source trajectory records.

## What Is Not Included

- Full raw datasets.
- Full processed tensors derived from raw datasets.
- Full training, validation, or test splits.
- Raw trajectory caches and map/image overlays used only for internal visual QA.
- Paper-scale checkpoints other than the single released reviewer metatype.

## User Responsibility

Users must obtain each dataset from the original provider and comply with all
dataset-specific license, citation, and redistribution requirements.
