# cultural_align Package

This directory contains the importable implementation.

Modules:

- `data`: dataset builders, schema definitions, and demo loaders.
- `models`: successor-feature and preference-vector models.
- `training`: training loops and transfer routines.
- `evaluation`: metrics and result aggregation.
- `utils`: shared configuration, logging, and reproducibility utilities.

The public model keeps the paper architecture fixed: temporal hidden dimension
`64` and Psi/Phi feature dimension `64`.
