import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    csv_file = "logs/Euler_Sequences_Debug.csv"
    
    if not os.path.exists(csv_file):
        print(f"Fehler: Die Datei '{csv_file}' existiert nicht.")
        print("Stelle sicher, dass du das Armeo-Tracking-Skript vorher mit dem 2D-Optimizer gestartet hast, damit die CSV generiert wird.")
        return

    # Daten laden
    print(f"Lese '{csv_file}'...")
    df = pd.read_csv(csv_file)
    
    if len(df) == 0:
        print("Die Datei ist leer.")
        return

    # Zeit normalisieren (Start bei 0 Sekunden)
    df['time'] = df['time'] - df['time'].iloc[0]

    sequences = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
    
    # Plot vorbereiten (6 Subplots für die 6 Sequenzen)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharex=True, sharey=False)
    axes = axes.flatten()
    
    fig.suptitle("Vergleich der Euler-Winkel-Sequenzen (Suche nach Sprüngen/Gimbal Lock)\nZ-Achse sollte um 0° bleiben (verbotene Achse)", fontsize=16, fontweight='bold')

    # Statistiken sammeln für späteren Vergleich
    stats_list = []

    for i, seq in enumerate(sequences):
        ax = axes[i]
        
        # Die Spaltennamen entsprechen z.B. xyz_1, xyz_2, xyz_3
        col1, col2, col3 = f"{seq}_1", f"{seq}_2", f"{seq}_3"
        
        ax.plot(df['time'], df[col1], label='Achse 1 (X)', linewidth=1.5, color='#1f77b4')
        ax.plot(df['time'], df[col2], label='Achse 2 (Y)', linewidth=1.5, color='#ff7f0e', alpha=0.9)
        ax.plot(df['time'], df[col3], label='Achse 3 (Z) [VERBOTEN]', linewidth=2, color='#d62728', alpha=0.8)
        
        # Statistiken berechnen
        vals = {
            'Achse 1': df[col1].values,
            'Achse 2': df[col2].values,
            'Achse 3 (Z)': df[col3].values
        }
        
        stats_seq = {'seq': seq}
        for axis_name, values in vals.items():
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            # Durchschnittliche Änderungsrate (absolut)
            diffs = np.abs(np.diff(values))
            avg_change_rate = np.mean(diffs) if len(diffs) > 0 else 0.0
            
            stats_seq[axis_name] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'avg_change': avg_change_rate
            }
        
        stats_list.append(stats_seq)
        
        # Statistik-Text im Plot anzeigen
        text_str = (
            f"Min/Max/Avg-Δ:\n"
            f"X: {vals['Achse 1'].min():.1f}°/{vals['Achse 1'].max():.1f}° (Δ:{np.mean(np.abs(np.diff(vals['Achse 1']))):.2f}°)\n"
            f"Y: {vals['Achse 2'].min():.1f}°/{vals['Achse 2'].max():.1f}° (Δ:{np.mean(np.abs(np.diff(vals['Achse 2']))):.2f}°)\n"
            f"Z: {vals['Achse 3 (Z)'].min():.1f}°/{vals['Achse 3 (Z)'].max():.1f}° (Δ:{np.mean(np.abs(np.diff(vals['Achse 3 (Z)']))):.2f}°)"
        )
        
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        ax.set_title(f"Sequenz: {seq.upper()}", fontweight='bold', fontsize=12)
        ax.set_ylabel("Winkel (°)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        if i >= 4:
            ax.set_xlabel("Zeit (s)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Konsolen-Ausgabe mit detaillierter Statistik
    print("\n" + "="*80)
    print("STATISTIK PRO SEQUENZ")
    print("="*80)
    
    for stat in stats_list:
        seq = stat['seq']
        print(f"\n>>> SEQUENZ: {seq.upper()}")
        print("-" * 50)
        for axis_name in ['Achse 1', 'Achse 2', 'Achse 3 (Z)']:
            s = stat[axis_name]
            print(f"  {axis_name:15} | Min: {s['min']:7.2f}°  Max: {s['max']:7.2f}°  "
                  f"Mittel: {s['mean']:7.2f}°  Avg-Δ: {s['avg_change']:6.2f}°/Δt")
    
    print("\n" + "="*80)
    print("BEWERTUNG DER Z-ACHSE (sollte minimal sein!):")
    print("="*80)
    for stat in stats_list:
        seq = stat['seq']
        z_range = stat['Achse 3 (Z)']['max'] - stat['Achse 3 (Z)']['min']
        z_avg_change = stat['Achse 3 (Z)']['avg_change']
        # Farb-Coding: Grün wenn klein, Rot wenn groß
        verdict = "✓ SEHR GUT" if z_range < 5 else "✓ GUT" if z_range < 15 else "⚠ MITTEL" if z_range < 30 else "✗ SCHLECHT"
        print(f"  {seq.upper():5} | Bereich: {z_range:6.2f}° | Δ-Rate: {z_avg_change:.2f}°/Δt | {verdict}")
    
    output_file = "logs/Euler_Sequences_Debug.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n✓ Plot gespeichert: {output_file}")
    print("Du kannst die Datei jetzt öffnen.")

if __name__ == "__main__":
    main()
