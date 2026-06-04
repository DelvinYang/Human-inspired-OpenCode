# Visualization Scripts

`plot_longtail_case_demo.R` recreates the three Appendix long-tail qualitative
case figures from the fixed derived data under `data/demo/longtail_cases/`.

Run from the repository root:

```bash
Rscript scripts/visualization/plot_longtail_case_demo.R
```

Optional environment variables:

```text
LONGTAIL_CASE_DEMO_DATA_DIR   Override the input data directory.
LONGTAIL_CASE_DEMO_OUT_DIR    Override the output directory.
```

Default output is written to `artifacts/longtail_case_demo/`.
