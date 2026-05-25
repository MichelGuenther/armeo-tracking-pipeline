import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

def evaluate_euler_sequences(csv_path, parent_cols, child_cols):
    """
    Lädt Sensor-Rohendaten, berechnet die relative Gelenkausrichtung und 
    testet alle 6 intrinsischen Euler-Sequenzen auf Sprünge (Gimbal Lock / Wrap-Around).
    """
    print(f"Lese Datei: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ Datei {csv_path} nicht gefunden!")
        print("Erstelle Dummy-Daten, um das Skript zu demonstrieren...")
        # Erstelle Dummy-Daten einer Bewegung, die nah an einer Singularität ist
        t = np.linspace(0, 10, 1000)
        # Parent ist statisch
        q_parent = np.tile([0, 0, 0, 1], (1000, 1))
        # Child macht eine problematische 2D Bewegung
        q_child = R.from_euler('xyz', np.column_stack([np.sin(t)*80, np.cos(t)*80, np.zeros_like(t)]), degrees=True).as_quat()
    else:
        df = pd.read_csv(csv_path)
        # Erwartet Spalten im Format [X, Y, Z, W] (SciPy Standard)
        # Passe dies an, falls dein raw-CSV ein anderes Format (z.B. [W, X, Y, Z]) hat!
        q_parent = df[parent_cols].to_numpy(dtype=float)
        q_child = df[child_cols].to_numpy(dtype=float)

    # 1. Relative Orientierung berechnen: R_rel = R_parent^-1 * R_child
    r_parent = R.from_quat(q_parent)
    r_child = R.from_quat(q_child)
    r_rel = r_parent.inv() * r_child

    # Alle 6 intrinsischen (mobile Achsen) Sequenzen
    sequences = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
    
    results = {}
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle('Euler-Winkel Verläufe für den 2D Joint (Sprung-Analyse)', fontsize=16)
    axes = axes.flatten()

    for idx, seq in enumerate(sequences):
        # 2. Winkel in Grad für die jeweilige Sequenz berechnen
        angles = r_rel.as_euler(seq, degrees=True)
        
        # 3. Sprünge detektieren (diff > 45 Grad zwischen zwei Frames ist verdächtig)
        diffs = np.abs(np.diff(angles, axis=0))
        # Summiere die "Ruckhaftigkeit" als Metrik auf
        jumps_count = np.sum(diffs > 45.0)
        total_variation = np.sum(diffs)
        
        results[seq] = {
            'jumps': jumps_count,
            'variation': total_variation
        }

        # 4. Plotten
        ax = axes[idx]
        ax.plot(angles[:, 0], label=f'Achse 1 ({seq[0].upper()})', alpha=0.8)
        ax.plot(angles[:, 1], label=f'Achse 2 ({seq[1].upper()})', alpha=0.8)
        ax.plot(angles[:, 2], label=f'Achse 3 ({seq[2].upper()})', alpha=0.8)
        
        ax.set_title(f"Sequenz: '{seq.upper()}' | Sprünge (>45°): {jumps_count}")
        ax.set_ylabel("Winkel (Grad)")
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 5. Auswertung in der Konsole
    print("\n" + "="*50)
    print("📊 ERGEBNIS DER SEQUENZ-ANALYSE")
    print("="*50)
    
    # Sortiere nach wenigsten Sprüngen, dann nach geringster Gesamtvariation
    best_seq = sorted(results.items(), key=lambda x: (x[1]['jumps'], x[1]['variation']))
    
    for rank, (seq, metrics) in enumerate(best_seq, 1):
        print(f"{rank}. '{seq.upper()}' -> Sprünge: {metrics['jumps']:5d} | Variation: {metrics['variation']:8.1f}°")
        
    print("\n💡 TIPP: Wähle die Sequenz aus Reihe 1. In der Anatomie (z.B. ISB Standard für die Schulter)")
    print("   ist meistens eine bestimmte Sequenz vorgegeben (z.B. YXY oder ZXY), um Gimbal Lock")
    print("   bei den normalen Bewegungen (Flexion, Abduktion) zu vermeiden.")
    
    output_png = "logs/Euler_Sequences_Comparison.png"
    plt.savefig(output_png, dpi=150)
    print(f"💾 Diagramm wurde unter '{output_png}' gespeichert!")
    
    plt.show()

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # KONFIGURATION DES SKRIPTS
    # -------------------------------------------------------------------------
    
    # 1. Trage hier den korrekten Pfad zu deiner aufgezeichneten CSV-Datei ein:
    MY_RAW_CSV_FILE = "logs/raw_sensor_recording_only2Dhorizontal.csv"
    
    # 2. Definiere die Spaltennamen für das Parent-Gelenk (Base / Torso):
    # SciPy R.from_quat() erwartet die Reihenfolge: [x, y, z, w]
    PARENT_QUAT_COLS = ["IMU_9e15c6_x", "IMU_9e15c6_y", "IMU_9e15c6_z", "IMU_9e15c6_w"]
    
    # 3. Definiere die Spaltennamen für das Child-Gelenk (Upper Arm):
    CHILD_QUAT_COLS = ["IMU_6dee46_x", "IMU_6dee46_y", "IMU_6dee46_z", "IMU_6dee46_w"]
    
    evaluate_euler_sequences(MY_RAW_CSV_FILE, PARENT_QUAT_COLS, CHILD_QUAT_COLS)
