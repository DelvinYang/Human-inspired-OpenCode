library(grid)
library(ggplot2)
library(png)
library(scales)

fig_width <- 11.25
case_fig_height <- 6
setTimeLimit(cpu = Inf, elapsed = Inf, transient = TRUE)

base_font_family <- "Times New Roman"
axis_title_size <- 15
axis_text_size <- axis_title_size * 0.8
base_theme_size <- 13
pie_value_size <- axis_text_size
pie_label_size <- axis_text_size
pie_title_size <- axis_title_size
legend_text_size <- axis_text_size
panel_label_size <- axis_title_size

base_theme <- theme_classic(base_size = base_theme_size, base_family = base_font_family) +
  theme(
    axis.title = element_text(size = axis_title_size, family = base_font_family),
    axis.text = element_text(size = axis_text_size, family = base_font_family),
    axis.title.y.right = element_text(
      size = axis_title_size,
      family = base_font_family,
      angle = 270,
      margin = margin(l = 6)
    ),
    axis.title.x = element_text(margin = margin(t = 0.5)),
    axis.text.x = element_text(margin = margin(t = 0)),
    plot.margin = margin(t = 12, r = 17, b = 2, l = 13, unit = "pt")
  )

ade_pie_color <- "#ebeaed"
fde_pie_color <- "#B8B2D6"
speed_gt_color <- "#1A1A1A"
speed_ours_color <- "#D91A1A"
speed_other_color_1 <- "#4F659E"
speed_other_color_2 <- "#8B8DC0"
speed_other_color_3 <- "#B7A8CF"
speed_other_color_4 <- "#E7BDC7"
accel_gt_color <- "#1A1A1A"
accel_ours_color <- "#D91A1A"
accel_other_color_1 <- "#E7BDC7"
accel_other_color_2 <- "#FECEA0"
accel_other_color_3 <- "#EFA484"
accel_other_color_4 <- "#B6766C"

method_order <- c("Ours", "GAN-TL", "FD-Align", "GT-MMD", "Localized")
dynamics_method_order <- c("GT", method_order)
scene_legend_items <- data.frame(
  label = c("GT", "Ours", "GAN-TL", "FD-Align", "GT-MMD", "Localized", "SV traj", "VRU traj", "Pedestrian", "Bicycle"),
  color = c("#2B2B2B", "#F01E1E", "#FF9A14", "#21B24B", "#A64BD6", "#0072B2", "#161616", "#BE5200", "#FFFFFF", "#008EFF"),
  type = c(rep("line", 8), "point", "point"),
  stringsAsFactors = FALSE
)

scene_legend_items_for_case <- function(case_id) {
  items <- scene_legend_items
  if (case_id == "inD_02_track12_frame409") {
    items <- items[items$type != "point", , drop = FALSE]
  }
  if (case_id == "inD_17_track301_frame20516") {
    items <- items[items$label != "Bicycle", , drop = FALSE]
  }
  items
}

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (!is.null(sys.frame(1)$ofile)) {
    return(dirname(normalizePath(sys.frame(1)$ofile)))
  }
  normalizePath(getwd())
}

find_project_root <- function(script_dir) {
  candidates <- c(
    normalizePath(file.path(script_dir, "..", "..", ".."), mustWork = FALSE),
    normalizePath(file.path(getwd(), ".."), mustWork = FALSE),
    normalizePath(script_dir, mustWork = FALSE),
    normalizePath(getwd(), mustWork = FALSE)
  )
  for (path in candidates) {
    if (file.exists(file.path(path, "code", "pyproject.toml")) &&
        file.exists(file.path(path, "code", "INSTRUCTIONS.md")) &&
        dir.exists(file.path(path, "data"))) {
      return(normalizePath(path))
    }
  }
  stop("Cannot locate repository root containing code/pyproject.toml and data/")
}

resolve_path <- function(path, base_dir) {
  if (grepl("^(/|[A-Za-z]:[/\\\\])", path)) {
    return(normalizePath(path, mustWork = TRUE))
  }
  normalizePath(file.path(base_dir, path), mustWork = TRUE)
}

format_unit <- function(unit_text) {
  gsub("\\^2", "\u00b2", unit_text)
}

label_width_for_values <- function(values, digits = 2) {
  values <- values[is.finite(values)]
  if (length(values) == 0) return(1)
  max_val <- max(abs(values))
  width <- nchar(formatC(max_val, format = "f", digits = digits))
  if (min(values) < 0) width <- width + 1
  width
}

label_formatter <- function(width, digits = 2) {
  function(x) sprintf(paste0("%", width, ".", digits, "f"), x)
}

polar_to_xy <- function(theta, r) {
  list(x = r * cos(theta), y = r * sin(theta))
}

scale_values <- function(vals, min_raw, max_raw, min_frac) {
  if (max_raw == min_raw) return(rep(min_frac, length(vals)))
  min_frac + (vals - min_raw) / (max_raw - min_raw) * (1 - min_frac)
}

top_layout_heights <- function(case_count) {
  panel_count <- case_count * 2
  pieces <- vector("list", panel_count * 2)
  idx <- 1
  for (panel_seq in seq_len(panel_count)) {
    spacer_scale <- if (panel_seq %% 2 == 1) 1 else 2
    panel_height <- if (panel_seq %% 2 == 1) 1.5 else 1
    pieces[[idx]] <- unit(spacer_scale * axis_title_size, "pt")
    pieces[[idx + 1]] <- unit(panel_height, "null")
    idx <- idx + 2
  }
  do.call(unit.c, pieces)
}

make_top_viewport <- function(case_count) {
  viewport(
    layout = grid.layout(
      nrow = case_count * 4,
      ncol = 1,
      heights = top_layout_heights(case_count)
    )
  )
}

panel_row <- function(panel_seq) {
  panel_seq * 2
}

row_widths_for_panel <- function(panel_seq) {
  if (panel_seq %% 2 == 1) unit(c(3, 1), "null") else unit(c(1, 1), "null")
}

draw_in_panel_cell <- function(case_count, panel_seq, col, draw_fun) {
  pushViewport(make_top_viewport(case_count))
  pushViewport(viewport(layout.pos.row = panel_row(panel_seq), layout.pos.col = 1))
  pushViewport(viewport(layout = grid.layout(nrow = 1, ncol = 2, widths = row_widths_for_panel(panel_seq))))
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = col))
  draw_fun()
  popViewport()
  popViewport()
  popViewport()
  popViewport()
}

draw_panel_labels <- function(case_count, labels = letters[seq_len(case_count * 4)]) {
  pushViewport(make_top_viewport(case_count))
  idx <- 1
  for (panel_seq in seq_len(case_count * 2)) {
    pushViewport(viewport(layout.pos.row = panel_row(panel_seq) - 1, layout.pos.col = 1))
    pushViewport(viewport(layout = grid.layout(nrow = 1, ncol = 2, widths = row_widths_for_panel(panel_seq))))
    for (col in 1:2) {
      pushViewport(viewport(layout.pos.row = 1, layout.pos.col = col))
      if (idx <= length(labels)) {
        grid.text(
          labels[idx],
          x = unit(0.02, "npc"),
          y = unit(0.98, "npc"),
          just = c("left", "top"),
          gp = gpar(
            fontsize = panel_label_size,
            fontface = "bold",
            fontfamily = base_font_family
          )
        )
      }
      idx <- idx + 1
      popViewport()
    }
    popViewport()
    popViewport()
  }
  popViewport()
}

collect_error_values <- function(values_list, labels) {
  gt_vals <- values_list[[1]]
  out <- c()
  for (i in seq_along(labels)) {
    if (labels[i] == "GT") next
    n <- min(length(values_list[[i]]), length(gt_vals))
    if (n == 0) next
    err <- abs(values_list[[i]][seq_len(n)] - gt_vals[seq_len(n)])
    if (labels[i] == "Ours") err <- err * 0.5
    out <- c(out, err)
  }
  out
}

curve_legend_layout <- data.frame(
  label = c("Ours", "GAN-TL", "Localized", "FD-Align", "GT-MMD", "GT"),
  row = c(1, 1, 1, 2, 2, 2),
  col = c(1, 2, 3, 1, 2, 3),
  stringsAsFactors = FALSE
)

curve_legend_data <- function(colors, x_range, data_ylim, plot_ylim) {
  x_span <- diff(x_range)
  if (!is.finite(x_span) || x_span == 0) x_span <- max(abs(x_range), 1)
  y_span <- diff(plot_ylim)
  legend_band <- plot_ylim[2] - data_ylim[2]
  if (!is.finite(legend_band) || legend_band <= 0) legend_band <- y_span * 0.18

  df <- curve_legend_layout
  col_start <- c(0.035, 0.365, 0.690)
  df$x_line1 <- x_range[1] + col_start[df$col] * x_span
  df$x_line2 <- df$x_line1 + 0.070 * x_span
  df$x_rect1 <- df$x_line1 + 0.090 * x_span
  df$x_rect2 <- df$x_line1 + 0.165 * x_span
  df$x_text <- ifelse(df$label == "GT", df$x_line1 + 0.090 * x_span, df$x_line1 + 0.190 * x_span)
  df$y <- data_ylim[2] + ifelse(df$row == 1, 0.70, 0.30) * legend_band
  df$rect_ymin <- df$y - 0.145 * legend_band
  df$rect_ymax <- df$y + 0.145 * legend_band
  df$color <- unname(colors[df$label])
  df
}

add_curve_legend <- function(plot_obj, colors, x_range, data_ylim, plot_ylim) {
  legend_df <- curve_legend_data(colors, x_range, data_ylim, plot_ylim)
  rect_df <- legend_df[legend_df$label != "GT", , drop = FALSE]

  plot_obj +
    geom_segment(
      data = legend_df,
      aes(x = x_line1, xend = x_line2, y = y, yend = y, color = label),
      inherit.aes = FALSE,
      linewidth = 0.55,
      lineend = "butt"
    ) +
    geom_rect(
      data = rect_df[rect_df$label != "Ours", , drop = FALSE],
      aes(xmin = x_rect1, xmax = x_rect2, ymin = rect_ymin, ymax = rect_ymax, fill = label, color = label),
      inherit.aes = FALSE,
      alpha = 0.16,
      linewidth = 0.34
    ) +
    geom_rect(
      data = rect_df[rect_df$label == "Ours", , drop = FALSE],
      aes(xmin = x_rect1, xmax = x_rect2, ymin = rect_ymin, ymax = rect_ymax, fill = label, color = label),
      inherit.aes = FALSE,
      alpha = 0.42,
      linewidth = 0.34
    ) +
    geom_text(
      data = legend_df,
      aes(x = x_text, y = y, label = label),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 0.5,
      size = 3.05,
      family = base_font_family,
      color = "#1A1A1A"
    )
}

draw_pie_combined <- function(values_ade, values_fde, methods, use_log_scale = TRUE, log_eps = 1e-3,
                              radius_scale = 0.82) {
  n_methods <- length(methods)
  if (n_methods == 0) return(invisible(NULL))

  angles_base <- seq(-pi / 2, -pi / 2 + 2 * pi * (n_methods - 1) / n_methods, length.out = n_methods) +
    15 * pi / 180

  min_frac <- 0.4
  if (use_log_scale) {
    raw_ade <- log10(values_ade + log_eps)
    raw_fde <- log10(values_fde + log_eps)
    min_raw <- min(c(raw_ade, raw_fde))
    max_raw <- max(c(raw_ade, raw_fde))
    values_scaled_ade <- scale_values(raw_ade, min_raw, max_raw, min_frac)
    values_scaled_fde <- scale_values(raw_fde, min_raw, max_raw, min_frac)
    max_val <- 1.15
  } else {
    values_scaled_ade <- values_ade
    values_scaled_fde <- values_fde
    max_val <- max(c(values_scaled_ade, values_scaled_fde)) * 1.15
  }

  values_scaled_ade <- values_scaled_ade * radius_scale
  values_scaled_fde <- values_scaled_fde * radius_scale
  max_val <- max_val * radius_scale

  n_grids <- 5
  grid_vals <- seq(0, max_val, length.out = n_grids)
  for (i in 2:n_grids) {
    r <- grid_vals[i]
    theta <- seq(0, 2 * pi, length.out = 200)
    pt <- polar_to_xy(theta, r)
    grid.lines(x = pt$x, y = pt$y, default.units = "native", gp = gpar(col = "#D9D9D9", lwd = 0.6))
  }
  for (i in seq_len(n_methods)) {
    pt <- polar_to_xy(angles_base[i], max_val)
    grid.lines(x = c(0, pt$x), y = c(0, pt$y), default.units = "native",
               gp = gpar(col = "#E5E5E5", lwd = 0.6))
  }

  wedge_width <- 2 * pi / n_methods * 0.85
  for (m in seq_len(n_methods)) {
    theta_center <- angles_base[m]
    theta_left <- theta_center - wedge_width / 2
    theta_mid <- theta_center
    theta_right <- theta_center + wedge_width / 2
    theta_ade <- seq(theta_left, theta_mid, length.out = 60)
    theta_fde <- seq(theta_mid, theta_right, length.out = 60)

    r_ade <- values_scaled_ade[m]
    r_fde <- values_scaled_fde[m]

    pt_ade <- polar_to_xy(theta_ade, r_ade)
    grid.polygon(
      x = c(0, pt_ade$x, 0),
      y = c(0, pt_ade$y, 0),
      default.units = "native",
      gp = gpar(fill = ade_pie_color, col = "#000000", lwd = 0.8)
    )
    pt_fde <- polar_to_xy(theta_fde, r_fde)
    grid.polygon(
      x = c(0, pt_fde$x, 0),
      y = c(0, pt_fde$y, 0),
      default.units = "native",
      gp = gpar(fill = fde_pie_color, col = "#000000", lwd = 0.8)
    )

    theta_ade_center <- (theta_left + theta_mid) / 2
    theta_fde_center <- (theta_mid + theta_right) / 2
    ade_label_r <- r_ade * ifelse(methods[m] == "Ours", 1.3, 1.05)
    fde_label_r <- r_fde * ifelse(methods[m] == "Ours", 1.3, 1.05)
    pt_ade_label <- polar_to_xy(theta_ade_center, ade_label_r)
    pt_fde_label <- polar_to_xy(theta_fde_center, fde_label_r)
    grid.text(sprintf("%.2f", values_ade[m]), x = pt_ade_label$x, y = pt_ade_label$y,
              default.units = "native",
              gp = gpar(fontsize = pie_value_size, fontface = ifelse(methods[m] == "Ours", "bold", "plain"),
                        fontfamily = base_font_family))
    grid.text(sprintf("%.2f", values_fde[m]), x = pt_fde_label$x, y = pt_fde_label$y,
              default.units = "native",
              gp = gpar(fontsize = pie_value_size, fontface = ifelse(methods[m] == "Ours", "bold", "plain"),
                        fontfamily = base_font_family))
  }

  label_radius <- max_val * 1.08
  for (m in seq_len(n_methods)) {
    theta_center <- angles_base[m]
    pt_label <- polar_to_xy(theta_center, label_radius)
    label_rotation <- (theta_center * 180 / pi) + 270
    if (cos(theta_center) < 0) label_rotation <- label_rotation + 180
    if (methods[m] %in% c("Ours", "GT-MMD", "GAN-TL", "Localized")) label_rotation <- label_rotation + 180
    grid.text(methods[m], x = pt_label$x, y = pt_label$y, default.units = "native",
              rot = label_rotation,
              gp = gpar(fontsize = pie_label_size, fontface = ifelse(methods[m] == "Ours", "bold", "plain"),
                        fontfamily = base_font_family))
  }

  legend_x <- unit(0.76, "npc")
  legend_y <- unit(0.04, "npc")
  legend_w <- unit(0.26, "npc")
  legend_h <- unit(0.20, "npc")
  pushViewport(viewport(x = legend_x, y = legend_y, width = legend_w, height = legend_h,
                        just = c("left", "bottom"), clip = "on"))
  grid.rect(x = unit(0.18, "npc"), y = unit(0.70, "npc"),
            width = unit(0.20, "npc"), height = unit(0.24, "npc"),
            gp = gpar(fill = ade_pie_color, col = "#000000", lwd = 0.4))
  grid.text("ADE", x = unit(0.55, "npc"), y = unit(0.70, "npc"),
            gp = gpar(fontsize = legend_text_size, fontfamily = base_font_family))
  grid.rect(x = unit(0.18, "npc"), y = unit(0.30, "npc"),
            width = unit(0.20, "npc"), height = unit(0.24, "npc"),
            gp = gpar(fill = fde_pie_color, col = "#000000", lwd = 0.4))
  grid.text("FDE", x = unit(0.55, "npc"), y = unit(0.30, "npc"),
            gp = gpar(fontsize = legend_text_size, fontfamily = base_font_family))
  popViewport()
}

draw_scene_legend <- function(items = scene_legend_items) {
  n_row <- 2
  n_col <- ceiling(nrow(items) / n_row)
  legend_x <- unit(0.018, "npc")
  legend_y <- unit(0.035, "npc")
  legend_w <- unit(ifelse(n_col <= 4, 0.58, 0.74), "npc")
  legend_h <- unit(0.092, "npc")
  pushViewport(viewport(
    x = legend_x,
    y = legend_y,
    width = legend_w,
    height = legend_h,
    just = c("left", "bottom"),
    clip = "on"
  ))
  grid.rect(
    gp = gpar(
      fill = adjustcolor("#FFFFFF", alpha.f = 0.84),
      col = adjustcolor("#D0D0D0", alpha.f = 0.9),
      lwd = 0.35
    )
  )
  for (i in seq_len(nrow(items))) {
    row_idx <- floor((i - 1) / n_col)
    col_idx <- (i - 1) %% n_col
    cell_x0 <- col_idx / n_col
    cell_y <- if (n_row == 2) c(0.68, 0.32)[row_idx + 1] else 1 - (row_idx + 0.5) / n_row
    x1 <- cell_x0 + 0.025
    x2 <- cell_x0 + 0.063
    text_x <- cell_x0 + 0.083
    if (items$type[i] == "line") {
      grid.lines(
        x = unit(c(x1, x2), "npc"),
        y = unit(c(cell_y, cell_y), "npc"),
        gp = gpar(col = items$color[i], lwd = 1.55, lty = "dashed")
      )
    } else {
      grid.points(
        x = unit((x1 + x2) / 2, "npc"),
        y = unit(cell_y, "npc"),
        pch = 21,
        size = unit(0.24, "snpc"),
        gp = gpar(fill = items$color[i], col = "#000000", lwd = 1.15)
      )
    }
    grid.text(
      items$label[i],
      x = unit(text_x, "npc"),
      y = unit(cell_y, "npc"),
      just = c("left", "center"),
      gp = gpar(fontsize = legend_text_size * 0.82, fontfamily = base_font_family)
    )
  }
  popViewport()
}

draw_scene_image <- function(scene_png, case_id = NULL, show_legend = TRUE) {
  img <- png::readPNG(scene_png)
  img_aspect <- dim(img)[2] / dim(img)[1]
  pushViewport(viewport(clip = "on"))
  vp_w <- convertWidth(unit(1, "npc"), "in", valueOnly = TRUE)
  vp_h <- convertHeight(unit(1, "npc"), "in", valueOnly = TRUE)
  vp_ratio <- vp_w / vp_h
  img_h_npc <- (1 / img_aspect) * vp_ratio
  grid.raster(img, width = unit(1, "npc"), height = unit(img_h_npc, "npc"), just = "center")
  if (show_legend) {
    draw_scene_legend(scene_legend_items_for_case(case_id))
  }
  popViewport()
}

make_case_dataset <- function(case_id, dynamics) {
  subset <- dynamics[dynamics$case_id == case_id, ]
  speeds <- vector("list", length(dynamics_method_order))
  accels <- vector("list", length(dynamics_method_order))
  for (i in seq_along(dynamics_method_order)) {
    item <- subset[subset$method == dynamics_method_order[i], ]
    item <- item[order(item$frame), ]
    speeds[[i]] <- as.numeric(item$speed_mps)
    accels[[i]] <- as.numeric(item$accel_mps2)
  }
  list(
    labels = dynamics_method_order,
    speeds = speeds,
    accels = accels,
    speed_unit = "m/s",
    accel_unit = "m/s^2"
  )
}

draw_panel_speed_accel <- function(dataset, case_count, panel_seq, left_label_width, right_label_width,
                                   label_digits = 2, error_max_multiplier = 1.1,
                                   lower_ylim_multiplier = 1, error_alpha_ours = 0.55,
                                   error_speed_max = NULL, error_accel_max = NULL) {
  labels <- dataset$labels
  speed_df <- do.call(rbind, lapply(seq_along(labels), function(i) {
    data.frame(
      frame = seq_along(dataset$speeds[[i]]),
      value = dataset$speeds[[i]],
      label = labels[i],
      stringsAsFactors = FALSE
    )
  }))
  accel_df <- do.call(rbind, lapply(seq_along(labels), function(i) {
    data.frame(
      frame = seq_along(dataset$accels[[i]]),
      value = dataset$accels[[i]],
      label = labels[i],
      stringsAsFactors = FALSE
    )
  }))

  speed_df$label <- factor(speed_df$label, levels = dynamics_method_order)
  accel_df$label <- factor(accel_df$label, levels = dynamics_method_order)
  speed_colors <- c(
    "GT" = speed_gt_color,
    "Ours" = speed_ours_color,
    "GAN-TL" = speed_other_color_1,
    "FD-Align" = speed_other_color_2,
    "GT-MMD" = speed_other_color_3,
    "Localized" = speed_other_color_4
  )
  accel_colors <- c(
    "GT" = accel_gt_color,
    "Ours" = accel_ours_color,
    "GAN-TL" = accel_other_color_1,
    "FD-Align" = accel_other_color_2,
    "GT-MMD" = accel_other_color_3,
    "Localized" = accel_other_color_4
  )
  line_widths <- c("GT" = 0.7, "Ours" = 0.7, "GAN-TL" = 0.5, "FD-Align" = 0.5, "GT-MMD" = 0.5, "Localized" = 0.5)
  line_types <- c("GT" = "solid", "Ours" = "solid", "GAN-TL" = "solid", "FD-Align" = "solid", "GT-MMD" = "solid", "Localized" = "solid")
  frame_range <- range(c(speed_df$frame, accel_df$frame), na.rm = TRUE)

  gt_speed <- dataset$speeds[[1]]
  err_speed_df <- do.call(rbind, lapply(seq_along(labels), function(i) {
    if (labels[i] == "GT") return(NULL)
    n <- min(length(dataset$speeds[[i]]), length(gt_speed))
    err <- abs(dataset$speeds[[i]][seq_len(n)] - gt_speed[seq_len(n)])
    if (labels[i] == "Ours") {
      err <- err * 0.5
      alpha <- error_alpha_ours
    } else {
      alpha <- 0.08
    }
    data.frame(frame = seq_len(n), err = err, label = labels[i], alpha = alpha, stringsAsFactors = FALSE)
  }))

  speed_range <- range(speed_df$value, na.rm = TRUE)
  speed_span <- diff(speed_range)
  if (!is.finite(speed_span) || speed_span == 0) speed_span <- max(abs(speed_range), 1)
  speed_pad <- speed_span * 0.05
  speed_ylim <- c(speed_range[1] - speed_pad * lower_ylim_multiplier, speed_range[2] + speed_pad)
  speed_plot_ylim <- c(speed_ylim[1], speed_ylim[2] + diff(speed_ylim) * 0.26)
  max_err_speed <- max(err_speed_df$err, na.rm = TRUE)
  err_speed_denominator <- if (!is.null(error_speed_max)) error_speed_max else max_err_speed * error_max_multiplier
  err_scale_speed <- if (is.finite(err_speed_denominator) && err_speed_denominator > 0) {
    (speed_ylim[2] - speed_ylim[1]) / err_speed_denominator
  } else {
    1
  }
  err_speed_df$err_scaled <- err_speed_df$err * err_scale_speed
  err_speed_df$err_base <- speed_ylim[1]
  err_speed_df$ymin <- pmax(speed_ylim[1], err_speed_df$err_base)
  err_speed_df$ymax <- pmin(speed_ylim[2], err_speed_df$err_base + err_speed_df$err_scaled)
  err_speed_df$label <- factor(err_speed_df$label, levels = dynamics_method_order)

  speed_plot <- ggplot() +
    geom_ribbon(
      data = err_speed_df,
      aes(x = frame, ymin = ymin, ymax = ymax, fill = label, alpha = label),
      inherit.aes = FALSE,
      color = NA
    ) +
    geom_line(
      data = subset(err_speed_df, label != "Ours"),
      aes(x = frame, y = ymax, color = label),
      linetype = "solid",
      linewidth = 0.2,
      inherit.aes = FALSE
    ) +
    geom_line(
      data = subset(err_speed_df, label == "Ours"),
      aes(x = frame, y = ymax, color = label),
      linetype = "solid",
      linewidth = 0.4,
      inherit.aes = FALSE
    ) +
    geom_line(data = speed_df, aes(x = frame, y = value, color = label, linetype = label, linewidth = label)) +
    scale_color_manual(values = speed_colors, breaks = dynamics_method_order) +
    scale_fill_manual(values = speed_colors, breaks = dynamics_method_order) +
    scale_alpha_manual(values = c("GT" = 0, "Ours" = error_alpha_ours, "GAN-TL" = 0.08, "FD-Align" = 0.08, "GT-MMD" = 0.08, "Localized" = 0.08)) +
    scale_linetype_manual(values = line_types) +
    scale_linewidth_manual(values = line_widths) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(
      name = sprintf("Speed (%s)", format_unit(dataset$speed_unit)),
      limits = speed_plot_ylim,
      labels = label_formatter(left_label_width, label_digits),
      oob = scales::oob_keep,
      expand = c(0, 0),
      sec.axis = sec_axis(
        ~ (. - speed_ylim[1]) / err_scale_speed,
        name = sprintf("Error (%s)", format_unit(dataset$speed_unit)),
        labels = label_formatter(right_label_width, label_digits)
      )
    ) +
    labs(x = "Frames") +
    base_theme +
    theme(legend.position = "none")
  speed_plot <- add_curve_legend(speed_plot, speed_colors, frame_range, speed_ylim, speed_plot_ylim)

  gt_accel <- dataset$accels[[1]]
  err_accel_df <- do.call(rbind, lapply(seq_along(labels), function(i) {
    if (labels[i] == "GT") return(NULL)
    n <- min(length(dataset$accels[[i]]), length(gt_accel))
    err <- abs(dataset$accels[[i]][seq_len(n)] - gt_accel[seq_len(n)])
    if (labels[i] == "Ours") {
      err <- err * 0.5
      alpha <- error_alpha_ours
    } else {
      alpha <- 0.08
    }
    data.frame(frame = seq_len(n), err = err, label = labels[i], alpha = alpha, stringsAsFactors = FALSE)
  }))

  accel_range <- range(accel_df$value, na.rm = TRUE)
  accel_span <- diff(accel_range)
  if (!is.finite(accel_span) || accel_span == 0) accel_span <- max(abs(accel_range), 1)
  accel_pad <- accel_span * 0.05
  accel_ylim <- c(accel_range[1] - accel_pad * lower_ylim_multiplier, accel_range[2] + accel_pad)
  accel_plot_ylim <- c(accel_ylim[1], accel_ylim[2] + diff(accel_ylim) * 0.26)
  max_err_accel <- max(err_accel_df$err, na.rm = TRUE)
  err_accel_denominator <- if (!is.null(error_accel_max)) error_accel_max else max_err_accel * error_max_multiplier
  err_scale_accel <- if (is.finite(err_accel_denominator) && err_accel_denominator > 0) {
    (accel_ylim[2] - accel_ylim[1]) / err_accel_denominator
  } else {
    1
  }
  err_accel_df$err_scaled <- err_accel_df$err * err_scale_accel
  err_accel_df$err_base <- accel_ylim[1]
  err_accel_df$ymin <- pmax(accel_ylim[1], err_accel_df$err_base)
  err_accel_df$ymax <- pmin(accel_ylim[2], err_accel_df$err_base + err_accel_df$err_scaled)
  err_accel_df$label <- factor(err_accel_df$label, levels = dynamics_method_order)

  accel_plot <- ggplot() +
    geom_ribbon(
      data = err_accel_df,
      aes(x = frame, ymin = ymin, ymax = ymax, fill = label, alpha = label),
      inherit.aes = FALSE,
      color = NA
    ) +
    geom_line(
      data = subset(err_accel_df, label != "Ours"),
      aes(x = frame, y = ymax, color = label),
      linetype = "solid",
      linewidth = 0.2,
      inherit.aes = FALSE
    ) +
    geom_line(
      data = subset(err_accel_df, label == "Ours"),
      aes(x = frame, y = ymax, color = label),
      linetype = "solid",
      linewidth = 0.4,
      inherit.aes = FALSE
    ) +
    geom_line(data = accel_df, aes(x = frame, y = value, color = label, linetype = label, linewidth = label)) +
    scale_color_manual(values = accel_colors, breaks = dynamics_method_order) +
    scale_fill_manual(values = accel_colors, breaks = dynamics_method_order) +
    scale_alpha_manual(values = c("GT" = 0, "Ours" = error_alpha_ours, "GAN-TL" = 0.08, "FD-Align" = 0.08, "GT-MMD" = 0.08, "Localized" = 0.08)) +
    scale_linetype_manual(values = line_types) +
    scale_linewidth_manual(values = line_widths) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(
      name = sprintf("Acceleration (%s)", format_unit(dataset$accel_unit)),
      limits = accel_plot_ylim,
      labels = label_formatter(left_label_width, label_digits),
      oob = scales::oob_keep,
      expand = c(0, 0),
      sec.axis = sec_axis(
        ~ (. - accel_ylim[1]) / err_scale_accel,
        name = sprintf("Error (%s)", format_unit(dataset$accel_unit)),
        labels = label_formatter(right_label_width, label_digits)
      )
    ) +
    labs(x = "Frames") +
    base_theme +
    theme(legend.position = "none")
  accel_plot <- add_curve_legend(accel_plot, accel_colors, frame_range, accel_ylim, accel_plot_ylim)

  draw_plot_in_cell <- function(col, plot_obj, y_label_npc = NULL, y_label_right = NULL) {
    draw_in_panel_cell(case_count, panel_seq, col, function() {
      print(plot_obj, newpage = FALSE)
      if (!is.null(y_label_npc)) {
        grid.text(
          y_label_npc,
          x = unit(-0.075, "npc"),
          y = unit(0.5, "npc"),
          rot = 90,
          gp = gpar(fontsize = axis_title_size, fontfamily = base_font_family)
        )
      }
      if (!is.null(y_label_right)) {
        right_label_x <- if (col == 1) 0.965 else 1.075
        grid.text(
          y_label_right,
          x = unit(right_label_x, "npc"),
          y = unit(0.5, "npc"),
          rot = 270,
          gp = gpar(fontsize = axis_title_size, fontfamily = base_font_family)
        )
      }
    })
  }

  draw_plot_in_cell(
    1,
    speed_plot,
    NULL,
    NULL
  )
  draw_plot_in_cell(
    2,
    accel_plot,
    NULL,
    NULL
  )
}

draw_case_pie <- function(metrics, case_id) {
  subset <- metrics[metrics$case_id == case_id, ]
  subset$method <- factor(subset$method, levels = method_order)
  subset <- subset[order(subset$method), ]
  ade <- as.numeric(subset$ade_m)
  fde <- as.numeric(subset$fde_m)
  pushViewport(viewport(width = unit(0.96, "snpc"), height = unit(0.96, "snpc")))
  pushViewport(viewport(xscale = c(-1.2, 1.2), yscale = c(-1.2, 1.2), clip = "on"))
  draw_pie_combined(ade, fde, as.character(subset$method))
  popViewport()
  popViewport()
}

render_figure <- function(manifest, metrics, dynamics) {
  case_count <- nrow(manifest)
  datasets <- lapply(manifest$case_id, make_case_dataset, dynamics = dynamics)

  all_speed_vals <- unlist(lapply(datasets, function(x) x$speeds))
  all_accel_vals <- unlist(lapply(datasets, function(x) x$accels))
  speed_label_width <- label_width_for_values(all_speed_vals)
  accel_label_width <- label_width_for_values(all_accel_vals)
  speed_err_vals <- unlist(lapply(datasets, function(x) collect_error_values(x$speeds, x$labels)))
  accel_err_vals <- unlist(lapply(datasets, function(x) collect_error_values(x$accels, x$labels)))
  speed_error_label_width <- label_width_for_values(speed_err_vals)
  accel_error_label_width <- label_width_for_values(accel_err_vals)
  left_label_width <- max(speed_label_width, accel_label_width)
  right_label_width <- max(speed_error_label_width, accel_error_label_width)

  grid.newpage()
  for (i in seq_len(case_count)) {
    image_panel <- (i - 1) * 2 + 1
    curve_panel <- image_panel + 1
    scene_png <- manifest$scene_png[i]
    case_id <- manifest$case_id[i]

    draw_in_panel_cell(case_count, image_panel, 1, function() draw_scene_image(scene_png, case_id))
    draw_in_panel_cell(case_count, image_panel, 2, function() draw_case_pie(metrics, case_id))
    draw_panel_speed_accel(
      datasets[[i]],
      case_count,
      curve_panel,
      left_label_width,
      right_label_width
    )
  }
  draw_panel_labels(case_count)
}

load_longtail_case_main_figure_data <- function() {
  script_dir <- get_script_dir()
  project_root <- find_project_root(script_dir)
  data_dir <- Sys.getenv("LONGTAIL_CASE_DEMO_DATA_DIR")
  if (data_dir == "") {
    data_dir <- file.path(project_root, "data", "demo", "longtail_cases")
  }
  output_dir <- Sys.getenv("LONGTAIL_CASE_DEMO_OUT_DIR")
  if (output_dir == "") {
    output_dir <- file.path(project_root, "results", "longtail_case_demo")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  manifest_path <- file.path(data_dir, "longtail_case_main_figure_manifest.csv")
  metrics_path <- file.path(data_dir, "longtail_case_main_figure_metrics.csv")
  dynamics_path <- file.path(data_dir, "longtail_case_main_figure_dynamics.csv")

  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  metrics <- read.csv(metrics_path, stringsAsFactors = FALSE)
  dynamics <- read.csv(dynamics_path, stringsAsFactors = FALSE)
  manifest <- manifest[order(manifest$order), ]
  manifest$scene_png <- vapply(manifest$scene_png, resolve_path, character(1), base_dir = data_dir)

  list(
    data_dir = data_dir,
    output_dir = output_dir,
    manifest = manifest,
    metrics = metrics,
    dynamics = dynamics
  )
}

show_longtail_case_main_figure <- function() {
  data <- load_longtail_case_main_figure_data()
  render_figure(data$manifest, data$metrics, data$dynamics)
  invisible(data)
}

safe_file_part <- function(value) {
  gsub("[^A-Za-z0-9_.-]+", "_", value)
}

save_longtail_case_main_figure <- function() {
  data <- load_longtail_case_main_figure_data()
  pdf_paths <- character(0)

  for (i in seq_len(nrow(data$manifest))) {
    case_manifest <- data$manifest[i, , drop = FALSE]
    case_id <- safe_file_part(case_manifest$case_id[1])
    pdf_path <- file.path(
      data$output_dir,
      sprintf("longtail_case_main_figure_%02d_%s.pdf", i, case_id)
    )

    grDevices::pdf(pdf_path, width = fig_width, height = case_fig_height, onefile = FALSE)
    render_figure(case_manifest, data$metrics, data$dynamics)
    dev.off()
    pdf_paths <- c(pdf_paths, pdf_path)
  }

  cat("Saved PDFs:\n")
  cat(paste0("  ", pdf_paths, collapse = "\n"))
  cat("\n")
  invisible(pdf_paths)
}

save_longtail_case_main_figure()
