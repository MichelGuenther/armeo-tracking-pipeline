import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot Optimizer debug logs showing points tested by least_squares.")
    parser.add_argument("file", nargs="?", default="logs/csv/debug_grid_search_1D.csv", 
                        help="Debug CSV-Datei mit getesteten Offsets")
    parser.add_argument("--window", type=int, help="Spezifischen Window-Index zum Plotten auswählen (zeigt nur die Cost-Landscape dieses Fensters detailliert)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Fehler: Datei {args.file} nicht gefunden.")
        sys.exit(1)

    df = pd.read_csv(args.file)
    
    if len(df) == 0:
        print("Datei ist leer.")
        sys.exit(1)

    if 'tested_yaw_deg' not in df.columns:
        print("Fehler: Das ist eine normale Log-Datei (z.B. session_XX_1D.csv), keine Debug-Datei! Bitte benutze hierfür 'plot_drift.py'.")
        sys.exit(1)

    # Zeit normalisieren (startet bei 0)
    df['time'] = df['time'] - df['time'].iloc[0]
    
    # Extrahiere eindeutige Window-Indizes
    windows = df['window_index'].unique()
    num_windows = len(windows)
    
    # Wenn ein spezifisches Fenster ausgewählt wurde, zeigen wir nur dieses im Detail an
    if args.window is not None:
        window_data = df[df['window_index'] == args.window].sort_values('tested_yaw_deg')
        if len(window_data) == 0:
            print(f"Fehler: Window {args.window} nicht in den Daten gefunden.")
            sys.exit(1)
        
        is_flat = window_data['is_flat_valley'].iloc[0] == 1
        if is_flat:
            print(f"Hinweis: Window {args.window} ist ein Flat Valley. Es wurden keine Tests durchgeführt.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Cost Landscape Detail - Window {args.window}", fontsize=14, fontweight='bold')
        
        if not is_flat:
            # We don't want lines bouncing around between stages, so just scatter the points
            ax.scatter(window_data['tested_yaw_deg'], window_data['cost_val'], 
                       s=30, color='tab:blue', alpha=0.6, label='Tested Points')
            
            # Find the absolute best point in this window
            best_idx = window_data['cost_val'].idxmin()
            best_pt = window_data.loc[best_idx]
            
            ax.scatter(best_pt['tested_yaw_deg'], best_pt['cost_val'], 
                      color='red', s=200, marker='*', edgecolors='darkred', linewidths=2, 
                      label='Global Minimum', zorder=10)
                
            ax.set_xlabel("Tested Yaw Offset [Grad]", fontweight='bold')
            ax.set_ylabel("Cost Value", fontweight='bold')
            ax.grid(True, alpha=0.5)
            
            # Add text box with window stats
            r_w_val = best_pt['r_w']
            is_singularity = "Yes" if r_w_val < 0.1 else "No"
            info_text = f"Window Quality (r_w): {r_w_val:.3f}\nSingularity Filter: {is_singularity}\nBest Cost: {best_pt['cost_val']:.4f}\nBest Yaw: {best_pt['tested_yaw_deg']:.2f}°"
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "FLAT VALLEY\n(Optimierung übersprungen)", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=20, color='red', fontweight='bold', transform=ax.transAxes)
            ax.axis('off')
            
        plt.tight_layout()
        import matplotlib
        if matplotlib.get_backend().lower() == "agg":
            output_path = os.path.splitext(args.file)[0] + f"_window_{args.window}_plot.png"
            if "csv" in output_path:
                output_path = output_path.replace("csv", "plots", 1)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Detail-Plot gespeichert nach: {output_path}")
        else:
            plt.show()

    # Plot erstellen (Übersicht)
    fig, axs = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(f"Optimizer Debug Analysis - Least Squares Explored Points\n({os.path.basename(args.file)})", 
                 fontsize=16, fontweight='bold')

    # 1. Optimization Landscape - Cost vs Tested Yaw
    ax = axs[0]
    for window_idx in windows[:min(5, num_windows)]:  # Zeige max 5 Fenster zum Vermeiden von Überzeichnung
        window_data = df[df['window_index'] == window_idx].sort_values('tested_yaw_deg')
        if len(window_data) > 0:
            ax.scatter(window_data['tested_yaw_deg'], window_data['cost_val'], 
                   s=15, alpha=0.6, label=f"Window {int(window_idx)}")
    
    # Markiere die besten Punkte
    # Group by window to find the absolute minimum for each
    best_points = df.loc[df.groupby('window_index')['cost_val'].idxmin()]
    if len(best_points) > 0:
        ax.scatter(best_points['tested_yaw_deg'], best_points['cost_val'], 
                  color='red', s=100, marker='*', edgecolors='darkred', linewidths=2, 
                  label='Best Found', zorder=10)
    
    ax.set_ylabel("Cost Value", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("(1) Optimization Landscape: Cost vs Tested Yaw Angles (Gauss-Newton Path)", fontweight='bold')

    # 2. All Tested Offsets with Cost Coloring
    ax = axs[1]
    scatter = ax.scatter(df['time'], df['tested_yaw_deg'], c=df['cost_val'], 
                        cmap='RdYlGn_r', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Markiere beste Punkte extra
    if len(best_points) > 0:
        ax.scatter(best_points['time'], best_points['tested_yaw_deg'], 
                  color='red', s=100, marker='*', edgecolors='darkred', linewidths=2, 
                  label='Best Found', zorder=10)
    
    ax.plot(df['time'], df['best_yaw_deg'], 'b-', linewidth=2, alpha=0.7, label='Current Best Yaw')
    ax.set_ylabel("Tested Yaw [°]", fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cost Value', fontweight='bold')
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("(2) All Tested Offsets (Color = Cost Value, Red Star = Best)", fontweight='bold')

    # Support column differences between 1D (up/low) and 2D (parent/child)
    parent_var_col = 'movement_var_parent' if 'movement_var_parent' in df.columns else 'movement_var_up'
    child_var_col = 'movement_var_child' if 'movement_var_child' in df.columns else 'movement_var_low'

    # 3. Movement Variance (Flat Valley Detection)
    ax = axs[2]
    ax.semilogy(df['time'], df[parent_var_col], label=f"Movement Variance ({parent_var_col})", 
               linewidth=1.5, color='tab:blue', marker='o', markersize=3)
    ax.semilogy(df['time'], df[child_var_col], label=f"Movement Variance ({child_var_col})", 
               linewidth=1.5, color='tab:orange', marker='s', markersize=3)
    
    # Markiere Flat Valley Events
    flat_valley_events = df[df['is_flat_valley'] == 1]
    if len(flat_valley_events) > 0:
        ax.scatter(flat_valley_events['time'], flat_valley_events[parent_var_col], 
                  color='red', s=100, marker='x', linewidth=2, label='Flat Valley Detected', zorder=10)
    
    ax.set_ylabel("Variance (log scale)", fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("(3) Movement Variance (Flat Valley Detection)", fontweight='bold')

    # 4. Window Quality (r_w) vs Flat Valley
    ax = axs[3]
    ax2 = ax.twinx()
    
    # Plot r_w auf linker Achse
    line1 = ax.plot(df['time'], df['r_w'], 'b-', linewidth=2, label="Window Quality ($r_w$)")
    ax.fill_between(df['time'], df['r_w'], alpha=0.2, color='tab:blue')
    
    # Plot Flat Valley auf rechter Achse
    flat_valley_numeric = df['is_flat_valley'].astype(float)
    line2 = ax2.plot(df['time'], flat_valley_numeric, 'r--', linewidth=2, label='Flat Valley Active')
    
    ax.set_ylabel("Quality Rating ($r_w$)", fontweight='bold', color='tab:blue')
    ax2.set_ylabel("Flat Valley (0/1)", fontweight='bold', color='red')
    ax.set_ylim([-0.05, 1.05])
    ax2.set_ylim([-0.1, 1.1])
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    ax.set_title("(4) Window Quality & Flat Valley Detection", fontweight='bold')

    # 5. Statistics per Window
    ax = axs[4]
    ax.axis('off')
    
    # Berechne Statistiken
    best_costs = best_points['cost_val']
    num_tested_per_window = df.groupby('window_index').size()
    num_flat_valleys = (df['is_flat_valley'] == 1).sum()
    
    stats_text = f"""
    OPTIMIZER (LEAST SQUARES) DEBUG STATISTICS
    ──────────────────────────────────────────────────────
    Total Test Points: {len(df)}
    Total Windows Processed: {num_windows}
    Avg Points per Window: {len(df) / num_windows:.1f}
    
    Cost Function:
      Overall Min Cost: {df['cost_val'].min():.6f}
      Overall Max Cost: {df['cost_val'].max():.6f}
      Overall Mean Cost: {df['cost_val'].mean():.6f}
      Std Dev Cost: {df['cost_val'].std():.6f}
    
    Tested Yaw Angles:
      Range: [{df['tested_yaw_deg'].min():.2f}°, {df['tested_yaw_deg'].max():.2f}°]
      Mean:  {df['tested_yaw_deg'].mean():.2f}°
    
    Best Yaw (Final):
      Mean: {df['best_yaw_deg'].mean():.2f}°
      Std:  {df['best_yaw_deg'].std():.2f}°
    
      Movement Analysis:
        Avg {parent_var_col}: {df[parent_var_col].mean():.6E}
        Avg {child_var_col}:  {df[child_var_col].mean():.6E}
        Flat Valley Events: {num_flat_valleys} ({100*num_flat_valleys/num_windows:.1f}% of windows)
    
    Window Quality:
      Avg r_w: {df['r_w'].mean():.4f}
      Min r_w: {df['r_w'].min():.4f}
      Max r_w: {df['r_w'].max():.4f}
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    import matplotlib
    if matplotlib.get_backend().lower() == "agg":
        output_path = os.path.splitext(args.file)[0] + "_debug_plot.png"
        if "csv" in output_path:
            output_path = output_path.replace("csv", "plots", 1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Debug-Plot gespeichert nach: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
