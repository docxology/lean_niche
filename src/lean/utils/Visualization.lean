/-!
# Mathematical Visualization and Plotting

This module provides visualization capabilities for mathematical concepts,
data plotting, and graphical representation of mathematical structures.
-/

import LeanNiche.Basic
import LeanNiche.Utils
import LeanNiche.Statistics

namespace LeanNiche.Visualization

open LeanNiche.Basic
open LeanNiche.Utils
open LeanNiche.Statistics

/-- Plot data structure -/
structure PlotData where
  title : String
  x_label : String
  y_label : String
  x_data : List Nat
  y_data : List Nat
  plot_type : PlotType

inductive PlotType where
  | line
  | scatter
  | bar
  | histogram
  | box
  | pie

/-- Visualization configuration -/
structure VisualizationConfig where
  width : Nat
  height : Nat
  dpi : Nat
  show_grid : Bool
  show_legend : Bool
  color_scheme : String
  font_size : Nat
  line_width : Nat

def default_viz_config : VisualizationConfig := {
  width := 800
  height := 600
  dpi := 96
  show_grid := true
  show_legend := true
  color_scheme := "viridis"
  font_size := 12
  line_width := 2
}

/-- ASCII art visualization functions -/
def create_ascii_plot (data : List Nat) (width : Nat) (height : Nat) : String :=
  if data.isEmpty then "No data to plot" else
    let max_val := list_max data
    let min_val := list_min data
    let range := max_val - min_val + 1
    let scale_factor := if range = 0 then 1 else height / range

    let scaled_data := data.map (λ x =>
      let normalized := if range = 0 then 0 else (x - min_val) * scale_factor
      Nat.min normalized height
    )

    let mut plot := ""
    for y in List.range (height + 1) do
      let row := List.range width |>.map (λ x =>
        if x < scaled_data.length then
          let data_y := scaled_data.get! x
          if data_y >= (height - y) then "*" else " "
        else " "
      )
      plot := plot ++ String.mk row ++ "\n"
    plot

def create_histogram_ascii (data : List Nat) (bins : Nat) : String :=
  if data.isEmpty then "No data for histogram" else
    let max_val := list_max data
    let min_val := list_min data
    let bin_width := (max_val - min_val) / bins
    let bin_counts := List.range bins |>.map (λ _ => 0)

    let histogram_data := data.foldl (λ counts x =>
      let bin_idx := if bin_width = 0 then 0 else (x - min_val) / bin_width
      let idx := Nat.min bin_idx (bins - 1)
      counts.set! idx (counts.get! idx + 1)
    ) bin_counts

    let max_count := list_max histogram_data
    let scale_factor := if max_count = 0 then 1 else 10 / max_count

    let mut histogram := "Histogram:\n"
    for i in List.range bins do
      let count := histogram_data.get! i
      let bar_length := count * scale_factor
      let bar := String.mk (List.replicate bar_length '*')
      let range_start := min_val + i * bin_width
      let range_end := min_val + (i + 1) * bin_width
      histogram := histogram ++ s!"{range_start}-{range_end}: {bar} ({count})\n"
    histogram

/-- Mathematical function plotting -/
def plot_function (f : Nat → Nat) (start : Nat) (end_ : Nat) (title : String) : String :=
  let x_values := List.range (end_ - start + 1) |>.map (λ x => x + start)
  let y_values := x_values.map f
  let data := PlotData.mk title "x" "f(x)" x_values y_values PlotType.line
  create_function_plot_string data

def create_function_plot_string (data : PlotData) : String :=
  let mut plot := s!"{data.title}\n"
  plot := plot ++ s!"{data.y_label}\n"
  plot := plot ++ "│\n"

  let max_y := list_max data.y_data
  let min_y := list_min data.y_data
  let y_range := max_y - min_y + 1

  for y in List.range (y_range + 1) do
    let current_y := max_y - y
    let y_marker := if current_y % 5 = 0 then nat_to_string current_y else " "
    plot := plot ++ s!"{y_marker} ┤ "

    for x in data.x_data do
      let y_val := data.y_data.get! (x - List.head! data.x_data)
      if y_val = current_y then "●" else " "
      plot := plot ++ (if y_val = current_y then "●" else " ")

    plot := plot ++ "\n"

  plot := plot ++ s!"└─{data.x_label}─→\n"
  plot := plot ++ s!"   {List.head! data.x_data} to {List.last! data.x_data}\n"
  plot

/-- Statistical visualization -/
def plot_distribution (data : List Nat) (title : String) : String :=
  let histogram := create_histogram_ascii data 10
  let stats := s!"Mean: {list_average data}, Median: {list_median data}, Max: {list_max data}, Min: {list_min data}"
  s!"{title}\n{stats}\n\n{histogram}"

def plot_probability_distribution (p : Nat) (trials : Nat) : String :=
  let outcomes := List.range 11  -- 0 to 10 successes
  let probabilities := outcomes.map (λ k =>
    if k <= trials then binomial trials k * (p ^ k) * ((1000 - p) ^ (trials - k)) / (1000 ^ trials) else 0
  )

  let mut plot := s!"Binomial Distribution (n={trials}, p={p/10}%)\n"
  plot := plot ++ "Successes | Probability\n"
  plot := plot ++ "──────────┼────────────\n"

  for i in List.range outcomes.length do
    let k := outcomes.get! i
    let prob := probabilities.get! i
    let bar_length := prob / 10  -- Scale for visualization
    let bar := String.mk (List.replicate bar_length '█')
    plot := plot ++ s!"{k:2}        │ {bar} {prob}‰\n"

  plot

/-- Dynamical systems visualization -/
def plot_trajectory (f : Nat → Nat) (x0 : Nat) (steps : Nat) : String :=
  let trajectory := List.range (steps + 1) |>.map (λ t =>
    if t = 0 then x0 else trajectory_helper f x0 t
  )

  let mut plot := s!"Trajectory of x_{0} = {x0}\n"
  plot := plot ++ "Time | State\n"
  plot := plot ++ "─────┼──────\n"

  for t in List.range (steps + 1) do
    let state := trajectory.get! t
    plot := plot ++ s!"{t:3}  │ {state:4}\n"

  plot

def trajectory_helper (f : Nat → Nat) (x : Nat) : Nat → Nat
  | 0 => x
  | n + 1 => f (trajectory_helper f x n)

/-- Lyapunov function visualization -/
def plot_lyapunov_function (V : Nat → Nat) (range : Nat) : String :=
  let values := List.range range |>.map V
  let data := PlotData.mk "Lyapunov Function V(x)" "x" "V(x)"
    (List.range range) values PlotType.line

  create_function_plot_string data

/-- Statistical data visualization -/
def plot_statistical_data (data : List Nat) (config : VisualizationConfig) : String :=
  let mean := list_average data
  let median := list_median data
  let max_val := list_max data
  let min_val := list_min data

  let mut plot := s!"Statistical Data Analysis\n"
  plot := plot ++ s!"Sample Size: {data.length}\n"
  plot := plot ++ s!"Mean: {mean}\n"
  plot := plot ++ s!"Median: {median}\n"
  plot := plot ++ s!"Maximum: {max_val}\n"
  plot := plot ++ s!"Minimum: {min_val}\n"
  plot := plot ++ s!"Range: {max_val - min_val}\n\n"

  plot := plot ++ create_ascii_plot data 60 20

  plot

/-- Box plot visualization -/
def create_box_plot (data : List Nat) : String :=
  if data.isEmpty then "No data for box plot" else
    let sorted := data.mergeSort (λ a b => a ≤ b)
    let n := sorted.length
    let q1 := sorted.get! (n / 4)
    let median := sorted.get! (n / 2)
    let q3 := sorted.get! (3 * n / 4)
    let iqr := q3 - q1
    let lower_fence := if q1 > 1.5 * iqr then q1 - (1.5 * iqr).toNat else list_min data
    let upper_fence := q3 + (1.5 * iqr).toNat

    let mut plot := "Box Plot:\n"
    plot := plot ++ s!"        ┌───┬────┬───┐\n"
    plot := plot ++ s!"        │ {lower_fence:2} │ {q1:2} │ {median:2} │ {q3:2} │ {upper_fence:2} │\n"
    plot := plot ++ s!"        └───┴────┴───┘\n"
    plot := plot ++ s!"Min: {list_min data}, Q1: {q1}, Median: {median}, Q3: {q3}, Max: {max_val}\n"
    plot := plot ++ s!"IQR: {iqr}, Lower Fence: {lower_fence}, Upper Fence: {upper_fence}\n"
    plot

/-- Export functions for different formats -/
def export_to_text (plot : String) (filename : String) : IO Unit := do
  IO.FS.writeFile filename plot

def export_to_csv (data : PlotData) (filename : String) : IO Unit := do
  let header := s!"{data.x_label},{data.y_label}\n"
  let rows := List.zip data.x_data data.y_data |>.map (λ (x, y) =>
    s!"{x},{y}\n"
  )
  let content := header ++ String.join rows
  IO.FS.writeFile filename content

def export_statistics (data : List Nat) (filename : String) : IO Unit := do
  let mean := list_average data
  let median := list_median data
  let max_val := list_max data
  let min_val := list_min data
  let stats := s!"Mean: {mean}\nMedian: {median}\nMaximum: {max_val}\nMinimum: {min_val}\nSample Size: {data.length}\n"
  IO.FS.writeFile filename stats

/-- Visualization gallery -/
def create_visualization_gallery : IO Unit := do
  IO.println "=== LeanNiche Visualization Gallery ==="

  -- Function plot example
  let func_plot := plot_function (λ x => x * x) 0 10 "Quadratic Function f(x) = x²"
  IO.println "\n1. Function Plot:"
  IO.println func_plot

  -- Statistical distribution example
  let stat_data := [1, 2, 3, 4, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 3, 2, 1, 4, 5, 6]
  let stat_plot := plot_statistical_data stat_data default_viz_config
  IO.println "\n2. Statistical Data Analysis:"
  IO.println stat_plot

  -- Probability distribution example
  let prob_plot := plot_probability_distribution 300 10  -- p=30%, n=10 trials
  IO.println "\n3. Probability Distribution:"
  IO.println prob_plot

  -- Trajectory example
  let traj_plot := plot_trajectory (λ x => (2 * x + 1) % 10) 3 15
  IO.println "\n4. Dynamical System Trajectory:"
  IO.println traj_plot

  IO.println "\n=== Visualization Gallery Complete ==="

end LeanNiche.Visualization
