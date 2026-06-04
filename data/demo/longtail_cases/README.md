# Long-Tail Case Figure Demo

This folder contains publication-safe derived data for recreating the three
long-tail qualitative case figures included in the Appendix.

The demo is fixed-data visualization only. It does not redistribute full inD raw
tracks, does not require model inference, and does not include comparison-method
checkpoints. The proposed-method and baseline trajectories are the saved
rollouts used for the Appendix figure.

Contents:

```text
longtail_case_main_figure_manifest.csv   Case order and scene-panel paths.
longtail_case_main_figure_metrics.csv    ADE/FDE values used by the polar panel.
longtail_case_main_figure_dynamics.csv   Speed and acceleration curves.
scene_panels/*.png                       Fixed scene panels with trajectory overlays.
```

Run from the repository root:

```bash
Rscript scripts/visualization/plot_longtail_case_demo.R
```

Expected output:

```text
artifacts/longtail_case_demo/longtail_case_main_figure_01_inD_02_track12_frame409.pdf
artifacts/longtail_case_demo/longtail_case_main_figure_02_inD_18_track94_frame5270.pdf
artifacts/longtail_case_demo/longtail_case_main_figure_03_inD_17_track301_frame20516.pdf
```

The script was tested with R 4.5.2 and the R packages `ggplot2`, `png`, and
`scales`.
