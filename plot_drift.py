import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
import matplotlib
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot heading drift from optimizer logs.")
    parser.add_argument("file", nargs="?", default="logs/csv/session_01_elbow.csv", help="Die CSV-Datei mit den Logs")
    parser.add_argument("--save", nargs="?", const=True, default=False, help="Speichere den Plot als PNG statt ihn anzuzeigen. Optional mit Zielpfad.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Fehler: Datei {args.file} nicht gefunden.")
        sys.exit(1)

    df = pd.read_csv(args.file)
    
    if len(df) == 0:
        print("Datei ist leer.")
        sys.exit(1)

    # If this is a debug grid-search CSV, suggest the appropriate plotter and exit
    if 'tested_yaw_deg' in df.columns:
        print("Eingabe sieht nach einem Debug-Grid-Search-Log aus. Verwende bitte plot_debug_grid_search.py für diese Datei.")
        sys.exit(0)

    # Zeit normalisieren (startet bei 0)
    df['time'] = df['time'] - df['time'].iloc[0]

    has_angle_y = 'angle_y' in df.columns
    has_angle_z = 'angle_z' in df.columns
    has_k_vals = 'k_b_w' in df.columns

    # Plot erstellen
    num_subplots = 7 if has_k_vals else 6
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 12 + (2 if has_k_vals else 0)), sharex=True)
    fig.suptitle(f"Heading Drift Analyse ({os.path.basename(args.file)})", fontsize=16)

    # Support different log schemas: prefer 'is_singular', fall back to debug names, else default False
    if 'is_singular' in df.columns:
        flat_valley_mask = df['is_singular'].astype(bool)
    elif 'is_flat_valley' in df.columns:
        flat_valley_mask = df['is_flat_valley'].astype(bool)
    elif 'is_flatvalley' in df.columns:
        flat_valley_mask = df['is_flatvalley'].astype(bool)
    else:
        flat_valley_mask = pd.Series(False, index=df.index)

    def mark_flat_valleys(ax):
        for st in df.loc[flat_valley_mask, 'time']:
            ax.axvline(st, color='red', alpha=0.08, linewidth=1)

    # 1. Heading Offsets (Raw + Bias + Filtered)
    axs[0].plot(df['time'], df['delta_w'], label=r"Roh-Messung Target ($\Delta_w$)", color='lightgray', linestyle='--')
    if 'b_w' in df.columns:
        axs[0].plot(df['time'], df['b_w'], label=r"Bias / Drift-Rate ($b_w$)", color='tab:orange', alpha=0.8)
    axs[0].plot(df['time'], df['delta_f_w'], label=r"Gefiltert ($\Delta_{f,w}$)", color='tab:blue', linewidth=2)
    
    # Flat Valley / Singularität markieren
    mark_flat_valleys(axs[0])
    
    axs[0].set_ylabel("Heading [Grad]")
    axs[0].grid(True)
    axs[0].legend(loc="upper left")

    # 2. Window Rating r_w
    axs[1].plot(df['time'], df['r_w'], label=r"Window Rating ($r_w$)", color='tab:green')
    axs[1].axhline(0.1, color='red', linestyle=':', label=r"Schwellenwert ($r_{min} = 0.1$)")
    axs[1].set_ylabel("Rating (0 bis 1)")
    axs[1].set_ylim([-0.05, 1.05])
    mark_flat_valleys(axs[1])
    axs[1].grid(True)
    axs[1].legend(loc="upper left")

    # 3. Bias Rate (Gier-Drift-Geschwindigkeit)
    axs[2].plot(df['time'], df['b_w'], label=r"Gelernte Drift-Rate ($b_w$)", color='tab:orange')
    axs[2].set_ylabel("Drift-Rate [Grad / Fenster]")
    mark_flat_valleys(axs[2])
    axs[2].grid(True)
    axs[2].legend(loc="upper left")

    # Globale Stetigkeit von b_w: finde ein Plateau, ab dem sich der Bias nur noch wenig ändert.
    if 'b_w' in df.columns:
        active_mask = ~flat_valley_mask
        bw_active = df.loc[active_mask, ['time', 'b_w']].copy()

        if len(bw_active) >= 6:
            plateau_window = max(5, min(25, len(bw_active) // 10))
            plateau_patience = max(3, plateau_window // 2)
            bw_range = bw_active['b_w'].max() - bw_active['b_w'].min()
            bw_tol = max(0.02, 0.05 * bw_range)

            bw_diff_mean = bw_active['b_w'].diff().abs().rolling(plateau_window, min_periods=plateau_window).mean()
            bw_value_std = bw_active['b_w'].rolling(plateau_window, min_periods=plateau_window).std()
            steady_candidates = (bw_diff_mean <= bw_tol) & (bw_value_std <= bw_tol)

            steady_start = None
            candidate_indices = steady_candidates[steady_candidates].index.to_list()
            for idx in candidate_indices:
                start_pos = bw_active.index.get_loc(idx)
                if start_pos + plateau_patience <= len(bw_active):
                    if steady_candidates.iloc[start_pos:start_pos + plateau_patience].all():
                        steady_start = idx
                        break

            if steady_start is not None:
                bw_steady = bw_active.loc[steady_start:, 'b_w']
            else:
                tail_len = max(5, len(bw_active) // 4)
                bw_steady = bw_active['b_w'].iloc[-tail_len:]

            bw_mean = bw_steady.mean()
            bw_std = bw_steady.std()
            settle_time = bw_active.loc[bw_steady.index[0], 'time'] if len(bw_steady) > 0 else np.nan
            steady_info = f"Bias-Plateau: t ≥ {settle_time:.1f}s\n$\\mu = {bw_mean:.4f}^\\circ / w$\n$\\sigma = {bw_std:.4f}^\\circ / w$"

            print(f"--- b_w Stetigkeit (globales Plateau) ---")
            print(f"Start Plateau:    t ≥ {settle_time:.2f}s")
            print(f"Mittelwert:       {bw_mean:.4f}° pro Fenster")
            print(f"Std-Abweichung:   {bw_std:.4f}° pro Fenster")
            print(f"(Je kleiner die Std-Abw., desto stetiger der Bias)\n")

            axs[2].text(0.01, 0.05, steady_info, transform=axs[2].transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Subplot: Filter Components Stacking Idea (instead of angles)
    # We calculate the raw correction step without the bias to show transparency
    diff_raw_filter = df['delta_w'] - df['delta_f_w']
    axs[3].plot(df['time'], diff_raw_filter, label=r"Abweichung: Roh - Gefiltert", color='tab:purple', alpha=0.8)
    axs[3].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs[3].fill_between(df['time'], 0, diff_raw_filter, color='tab:purple', alpha=0.2)
                        
    axs[3].set_ylabel("Diff (Roh - Filter) [Grad]")
    mark_flat_valleys(axs[3])
    axs[3].grid(True)
    axs[3].legend(loc="upper left")

    # 5. Flat Valley / Singularity Status
    axs[4].fill_between(df['time'], flat_valley_mask.astype(float), color='red', alpha=0.25, step='mid', label="Flat Valley aktiv")
    axs[4].plot(df['time'], flat_valley_mask.astype(float), color='darkred', linewidth=1.5)
    axs[4].set_ylabel("Flat Valley")
    axs[4].set_ylim([-0.05, 1.05])
    axs[4].set_yticks([0, 1])
    axs[4].set_yticklabels(["aus", "an"])
    axs[4].grid(True)
    # 6. Joint Angles
    axs[5].plot(df['time'], df['angle_x'], label="Angle X (Flexion)", color='tab:blue')
    if has_angle_y:
        axs[5].plot(df['time'], df['angle_y'], label="Angle Y (Abduction)", color='tab:green')
    if has_angle_z:
        axs[5].plot(df['time'], df['angle_z'], label="Angle Z (Rotation)", color='tab:red')
    axs[5].set_ylabel("Angles [Grad]")
    mark_flat_valleys(axs[5])
    axs[5].grid(True)

    # --- NEU: Konvergenz berechnen und Min/Max markieren ---
    # Konvergenz: Wir prüfen, wann der gleitende Durchschnitt (10 Fenster) 
    # der Abweichung zwischen Roh-Offset und gefiltertem Offset unter 2 Grad fällt.
    diff_deg = (df['delta_w'] - df['delta_f_w']).abs()
    diff_smooth = diff_deg.rolling(window=10, min_periods=1).mean()
    
    conv_mask = diff_smooth < 2.0
    if conv_mask.any():
        conv_idx = conv_mask.idxmax()
        conv_time = df.loc[conv_idx, 'time']
    else:
        conv_idx = df.index[0]
        conv_time = df['time'].iloc[0]
        
    axs[5].axvline(conv_time, color='purple', linestyle='--', linewidth=2, label="Filter Konvergenz")
    
    # Betrachtet werden nur noch Daten NACH der Konvergenz
    df_conv = df.loc[conv_idx:]
    
    if len(df_conv) > 0:
        rom_info = ""
        
        def mark_min_max(col_name, color, label_prefix):
            nonlocal rom_info
            mi_idx, ma_idx = df_conv[col_name].idxmin(), df_conv[col_name].idxmax()
            mi, ma = df_conv.loc[mi_idx, col_name], df_conv.loc[ma_idx, col_name]
            t_mi, t_ma = df_conv.loc[mi_idx, 'time'], df_conv.loc[ma_idx, 'time']
            
            axs[5].scatter([t_mi, t_ma], [mi, ma], color=color, marker='*', s=150, zorder=5, edgecolor='black')
            
            # Beschriftung leicht versetzt anbringen
            axs[5].text(t_mi, mi - (5 if mi < 0 else -5), f"{mi:.1f}°", ha='center', va=('top' if mi < 0 else 'bottom'), color=color, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            axs[5].text(t_ma, ma + (5 if ma > 0 else -5), f"{ma:.1f}°", ha='center', va=('bottom' if ma > 0 else 'top'), color=color, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            
            rom_info += f"{label_prefix}: [{mi:.1f}°, {ma:.1f}°]\n"

        mark_min_max('angle_x', 'tab:blue', 'RoM X')
        
        if has_angle_y:
            mark_min_max('angle_y', 'tab:green', 'RoM Y')
        if has_angle_z:
            mark_min_max('angle_z', 'tab:red', 'RoM Z')
            
        print(f"\n--- Range of Motion (ab Konvergenz t>={conv_time:.1f}s) ---\n{rom_info}")
    else:
        rom_info = "Warte auf Konvergenz...\n"

    axs[5].legend(loc="upper left")
    axs[5].text(0.01, 0.05, f"Post-Konvergenz:\n{rom_info.strip()}", transform=axs[5].transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 7. Adaptive Filter Gains (k_b_w und k_delta_w)
    if has_k_vals:
        # Falls 7 Subplots existieren
        axs[-1].plot(df['time'], df['k_delta_w'], label=r"$k_{\Delta w}$ (Heading Gain)", color='tab:blue', linewidth=2)
        axs[-1].plot(df['time'], df['k_b_w'], label=r"$k_{b\,w}$ (Bias Gain)", color='tab:orange', linewidth=2)
        axs[-1].set_ylabel("Filter Gains (k)")
        axs[-1].set_xlabel("Zeit [Sekunden]")
        mark_flat_valleys(axs[-1])
        axs[-1].grid(True)
        axs[-1].legend(loc="upper right")
    else:
        axs[5].set_xlabel("Zeit [Sekunden]")

    plt.tight_layout()
    if args.save is not False or matplotlib.get_backend().lower() == "agg":
        output_path = args.save if isinstance(args.save, str) else os.path.splitext(args.file)[0] + "_plot.png"
        # If output_path is in logs/csv, redirect to logs/plots
        if "csv" in output_path:
            output_path = output_path.replace("csv", "plots", 1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert nach: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
