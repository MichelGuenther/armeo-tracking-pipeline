import os
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'paper']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'black']

def main():
    data = {}
    for m in METHODS:
        fpath = f"logs/euler_eval_{m}.csv"
        if os.path.exists(fpath):
            data[m] = pd.read_csv(fpath)
        else:
            print(f"Warnung: {fpath} nicht gefunden.")

    if not data:
        print("Keine Daten zum Plotten gefunden!")
        return

    # --- PLOT 1: Delta W (Heading) ---
    plt.figure(figsize=(12, 6))
    for i, m in enumerate(data.keys()):
        df = data[m]
        # Hebe die Paper-Methode dicker hervor
        linewidth = 2.5 if m == 'paper' else 1.5
        linestyle = '-' if m == 'paper' else '--'
        plt.plot(df['time'], df['delta_w_deg'], label=f"Method: {m}", color=COLORS[i], linewidth=linewidth, linestyle=linestyle, alpha=0.8)
    
    plt.title("Vergleich der Heading-Schätzung (Delta W) für verschiedene Euler-Sequenzen")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Delta W [Grad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/Euler_Comparison_DeltaW.png", dpi=150)
    print("Gespeichert: logs/Euler_Comparison_DeltaW.png")
    
    # --- PLOT 2: Gelenkwinkel (Flexion & Abduktion) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for i, m in enumerate(data.keys()):
        df = data[m]
        linewidth = 2.5 if m == 'paper' else 1.5
        linestyle = '-' if m == 'paper' else '--'
        
        ax1.plot(df['time'], df['angle_x_deg'], label=f"{m}", color=COLORS[i], linewidth=linewidth, linestyle=linestyle, alpha=0.8)
        ax2.plot(df['time'], df['angle_y_deg'], label=f"{m}", color=COLORS[i], linewidth=linewidth, linestyle=linestyle, alpha=0.8)

    ax1.set_title("Angle X (Flexion) - Vergleichende Sprunganalyse")
    ax1.set_ylabel("Grad")
    ax1.grid(True)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax2.set_title("Angle Y (Abduction) - Vergleichende Sprunganalyse")
    ax2.set_xlabel("Zeit [s]")
    ax2.set_ylabel("Grad")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("logs/Euler_Comparison_Angles.png", dpi=150)
    print("Gespeichert: logs/Euler_Comparison_Angles.png")

if __name__ == "__main__":
    main()
