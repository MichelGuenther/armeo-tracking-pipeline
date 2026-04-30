import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Plot heading drift from optimizer logs.")
    parser.add_argument("file", nargs="?", default="logs/session_01_elbow.csv", help="Die CSV-Datei mit den Logs")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Fehler: Datei {args.file} nicht gefunden.")
        sys.exit(1)

    df = pd.read_csv(args.file)
    
    if len(df) == 0:
        print("Datei ist leer.")
        sys.exit(1)

    # Zeit normalisieren (startet bei 0)
    df['time'] = df['time'] - df['time'].iloc[0]

    # Plot erstellen
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Heading Drift Analyse ({args.file})", fontsize=16)

    # 1. Heading Offsets (Roh / Gefiltert)
    axs[0].plot(df['time'], df['delta_w'], label="Roh-Messung ($\delta_w$)", color='lightgray', linestyle='--')
    axs[0].plot(df['time'], df['delta_f_w'], label="Gefiltert ($\delta_{f,w}$)", color='tab:blue', linewidth=2)
    
    # Singularitäten markieren
    singular_times = df[df['is_singular'] == 1]['time']
    for st in singular_times:
        axs[0].axvline(st, color='red', alpha=0.1)
    
    axs[0].set_ylabel("Heading Offset [Grad]")
    axs[0].grid(True)
    axs[0].legend(loc="upper left")

    # 2. Window Rating r_w
    axs[1].plot(df['time'], df['r_w'], label="Window Rating ($r_w$)", color='tab:green')
    axs[1].axhline(0.1, color='red', linestyle=':', label="Schwellenwert ($r_{min} = 0.1$)")
    axs[1].set_ylabel("Rating (0 bis 1)")
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].grid(True)
    axs[1].legend(loc="upper left")

    # 3. Bias Rate (Gier-Drift-Geschwindigkeit)
    axs[2].plot(df['time'], df['b_w'], label="Gelernte Drift-Rate ($b_w$)", color='tab:orange')
    axs[2].set_ylabel("Drift-Rate [Grad / Fenster]")
    axs[2].grid(True)
    axs[2].legend(loc="upper left")

    # 4. Gelenkwinkel / Flexion
    if 'angle_x' in df.columns:
        axs[3].plot(df['time'], df['angle_x'], label="Gelenk-Winkel (X)", color='tab:purple')
    if 'angle_y' in df.columns:
        axs[3].plot(df['time'], df['angle_y'], label="Gelenk-Winkel (Y)", color='tab:cyan', alpha=0.7)
    if 'angle_z' in df.columns:
        axs[3].plot(df['time'], df['angle_z'], label="Gelenk-Winkel (Z)", color='tab:pink', alpha=0.7)
        
    axs[3].set_ylabel("Winkel [Grad]")
    axs[3].set_xlabel("Zeit [Sekunden]")
    axs[3].grid(True)
    axs[3].legend(loc="upper left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
