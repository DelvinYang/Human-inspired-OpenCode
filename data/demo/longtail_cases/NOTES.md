# Long-Tail Case Figure Demo

This folder contains the inputs for the three long-tail qualitative case figures
included in the Appendix.

Contents:

```text
longtail_case_main_figure_manifest.csv   Case order and scene-panel paths.
longtail_case_main_figure_metrics.csv    ADE/FDE values used by the polar panel.
longtail_case_main_figure_dynamics.csv   Speed and acceleration curves.
scene_panels/*.png                       Scene panels with trajectory overlays.
```

Run from the repository root:

```bash
Rscript code/scripts/visualization/plot_longtail_case_demo.R
```

Expected output:

```text
results/longtail_case_demo/longtail_case_main_figure_01_inD_02_track12_frame409.pdf
results/longtail_case_demo/longtail_case_main_figure_02_inD_18_track94_frame5270.pdf
results/longtail_case_demo/longtail_case_main_figure_03_inD_17_track301_frame20516.pdf
```

The script was tested with R 4.5.2 and the R packages `ggplot2`, `png`, and
`scales`.
