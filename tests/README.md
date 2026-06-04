# Tests

Tests use only the small inD demo subset.

Run them from the repository root:

```bash
PYTHONPATH=src python -m unittest discover tests
```

The default tests check package imports, fixed model dimensions, released
checkpoint loading, and the demo NPZ schema against the expected summary.
Full-dataset tests should not be part of default CI.
