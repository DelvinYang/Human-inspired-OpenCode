# Data Policy

This release intentionally does not include full third-party driving datasets.
The paper uses public or access-controlled trajectory datasets whose providers
retain their own redistribution rules. Publishing full raw or processed copies
inside this repository would create license and attribution risk.

## What Is Included

- A small inD demo/test subset, after confirming it is compliant with the
  applicable provider terms.
- Code and scripts for all supported datasets.
- Configuration templates that point users to local dataset locations.

## What Is Not Included

- Full raw datasets.
- Full processed tensors derived from raw datasets.
- Full training, validation, or test splits.
- Checkpoints trained on restricted third-party data, unless separately cleared.

## User Responsibility

Users must obtain each dataset from the original provider and comply with all
dataset-specific license, citation, and redistribution requirements.
