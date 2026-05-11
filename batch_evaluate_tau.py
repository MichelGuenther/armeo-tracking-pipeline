import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from scipy.spatial.transform import Rotation as R
from scipy.stats import linregress

from optimizer import Optimizer1D, Optimizer2D_Universal

# ==============================================================================
# --- 1. KONFIGURATION FÜR DIE EVALUATION ---
# ==============================================================================

# Die Raw-Sensor-Aufnahme, die als Grundlage dient
RAW_CSV_FILE = "logs/raw_sensor_recording_long.csv"  # Ggf. anpassen (z.B. raw_sensor_recording_2.csv)

# Welche Tau-Werte sollen gegeneinander getestet werden?
TAU_VALUES_TO_TEST = [1.0, 5.0, 15.0, 30.0]

OUTPUT_DIR = "logs/tau_evaluation_6_deltadelta_noflatvalley_long"  # Alle Ergebnisse werden hier gespeichert (CSV + Diagramm)

# Sensor IDs (müssen mit der CSV übereinstimmen)
ID_BASE = 'IMU_9e15c6'
ID_UPPER = 'IMU_6dee46'
ID_LOWER = 'IMU_c22f23'

# Parameter aus der Live-Bridge
SENSOR_HZ = 200
DATA_WINDOW_SEC = 1
CALCULATION_INTERVAL_SEC = 0.25
OPT_WINDOW_SIZE = int(SENSOR_HZ * DATA_WINDOW_SEC)
OPT_STEP_SIZE = int(SENSOR_HZ * CALCULATION_INTERVAL_SEC)
OPT_FLAT_VALLEY_THRESHOLD = 1e-7
OPT_DELTA_DELTA_WEIGHT = OPT_WINDOW_SIZE / math.pi

# --- FILTER-KONFIGURATION: Hier kannst du die Filterfeatures anpassen ---
ENABLE_SINGULARITY = True
ENABLE_FLAT_VALLEY = True
ENABLE_ANTI_WINDUP = True
ENABLE_LIMROM_2D = True
DELTA_DELTA_WEIGHT = OPT_DELTA_DELTA_WEIGHT

# Kalibrierung aus test_bridge.py
R_ALIGN_BASE =  R.from_euler('xyz', [-90, 0, 0], degrees=True)
R_ALIGN_UPPER = R.from_euler('xyz', [-90, 0, 0], degrees=True)
R_ALIGN_LOWER = R.from_euler('xyz', [-90, 0, 180], degrees=True)

Q_MAP_BASE  = np.array([ 1, -1, -1, 1], dtype=np.float32) 
Q_MAP_UPPER = np.array([ 1, -1, -1, 1], dtype=np.float32)
Q_MAP_LOWER = np.array([-1,  1, -1, 1], dtype=np.float32)

# ==============================================================================
# --- 2. SYNCHRONE OPTIMIZER FÜR OFFLINE-BATCH-VERARBEITUNG ---
# ==============================================================================
# Wir überschreiben die Threading-Logik, damit wir die CSV-Datei in maximaler
# Geschwindigkeit abarbeiten können, ohne dass Frames übersprungen werden.

class SyncOptimizer1D(Optimizer1D):
    def add_packet_and_optimize(self, r_up_aligned, r_low_aligned):
        self.buffer_upper.append(r_up_aligned.as_quat())
        self.buffer_lower.append(r_low_aligned.as_quat())
        
        if len(self.buffer_upper) >= self.window_size:
            buf_up_copy = self.buffer_upper.copy()
            buf_low_copy = self.buffer_lower.copy()
            
            # Synchroner Aufruf (blockiert, bis fertig gerechnet)
            self._run_optimization_threaded(buf_up_copy, buf_low_copy)
            
            keep_elements = max(0, self.window_size - self.step_size)
            self.buffer_upper = self.buffer_upper[-keep_elements:] if keep_elements > 0 else []
            self.buffer_lower = self.buffer_lower[-keep_elements:] if keep_elements > 0 else []

class SyncOptimizer2D(Optimizer2D_Universal):
    def add_packet_and_optimize(self, r_par_aligned, r_chi_aligned):
        self.buffer_parent.append(r_par_aligned.as_quat())
        self.buffer_child.append(r_chi_aligned.as_quat())
        
        if len(self.buffer_parent) >= self.window_size:
            buf_par_copy = self.buffer_parent.copy()
            buf_chi_copy = self.buffer_child.copy()
            
            # Synchroner Aufruf
            self._run_optimization_threaded(buf_par_copy, buf_chi_copy)
            
            keep_elements = max(0, self.window_size - self.step_size)
            self.buffer_parent = self.buffer_parent[-keep_elements:] if keep_elements > 0 else []
            self.buffer_child = self.buffer_child[-keep_elements:] if keep_elements > 0 else []

# ==============================================================================
# --- 3. HAUPTSKRIPT: BERECHNUNG & PLOTTING ---
# ==============================================================================

def main():
    if not os.path.exists(RAW_CSV_FILE):
        print(f"❌ Fehler: Konnte '{RAW_CSV_FILE}' nicht finden.")
        print("Bitte passe den Pfad in RAW_CSV_FILE oben im Skript an eine deiner Aufnahmen an.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prüfe, ob bereits CSVs vorhanden sind
    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
    if csv_files:
        print(f"📁 Gefunden: {len(csv_files)} CSV-Dateien in {OUTPUT_DIR}. Überspringe Berechnungen und gehe direkt zum Plotten.")
        plot_only = True
        df_raw = None  # Nicht benötigt
    else:
        plot_only = False
        print(f"📂 Lese Sensor-Daten aus {RAW_CSV_FILE} (kann einen Moment dauern)...")
        df_raw = pd.read_csv(RAW_CSV_FILE)
        total_rows = len(df_raw)
        print(f"✅ {total_rows} Zeilen geladen. Berechne {len(TAU_VALUES_TO_TEST)} Durchläufe...\n")

    if not plot_only:
        for tau_d in TAU_VALUES_TO_TEST:
            for tau_b in TAU_VALUES_TO_TEST:
                log_1d = os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_1D.csv")
                log_2d = os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_2D.csv")
                
                opt_1d = SyncOptimizer1D(
                    sensor_upper=ID_UPPER, sensor_lower=ID_LOWER, 
                    window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
                    flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
                    enable_singularity_filter=ENABLE_SINGULARITY, enable_flat_valley_filter=ENABLE_FLAT_VALLEY, enable_anti_windup=ENABLE_ANTI_WINDUP, tau_b_=tau_b, tau_delta_=tau_d, delta_delta_weight=DELTA_DELTA_WEIGHT,
                    log_file=log_1d, debug_log_file=None # Debug Log aus, da es das Batching extrem verlangsamt
                )
                
                opt_2d = SyncOptimizer2D(
                    sensor_parent=ID_BASE, sensor_child=ID_UPPER, 
                    window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
                    flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD, delta_delta_weight=OPT_DELTA_DELTA_WEIGHT,
                    enable_singularity_filter=ENABLE_SINGULARITY, enable_flat_valley_filter=ENABLE_FLAT_VALLEY, enable_anti_windup=ENABLE_ANTI_WINDUP, enable_limrom=ENABLE_LIMROM_2D, tau_b_=tau_b, tau_delta_=tau_d, 
                    log_file=log_2d, debug_log_file=None
                )
                
                print(f"🚀 Starte Verarbeitung für tau_b = {tau_b}, tau_delta = {tau_d}...")
                
                # Unterdrücke den massiven Terminal-Spam der Optimizer während der Batch-Verarbeitung
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        for idx, row in df_raw.iterrows():
                            # Quaternionen auslesen (Fallback auf Identity, falls fehlerhaft)
                            q_b = [row.get(f'{ID_BASE}_w', 1), row.get(f'{ID_BASE}_x', 0), row.get(f'{ID_BASE}_y', 0), row.get(f'{ID_BASE}_z', 0)]
                            q_u = [row.get(f'{ID_UPPER}_w', 1), row.get(f'{ID_UPPER}_x', 0), row.get(f'{ID_UPPER}_y', 0), row.get(f'{ID_UPPER}_z', 0)]
                            q_l = [row.get(f'{ID_LOWER}_w', 1), row.get(f'{ID_LOWER}_x', 0), row.get(f'{ID_LOWER}_y', 0), row.get(f'{ID_LOWER}_z', 0)]
                            
                            # SciPy Format [x,y,z,w] & Alignment
                            r_base = R.from_quat(np.array([q_b[1], q_b[2], q_b[3], q_b[0]]) * Q_MAP_BASE) * R_ALIGN_BASE
                            r_up   = R.from_quat(np.array([q_u[1], q_u[2], q_u[3], q_u[0]]) * Q_MAP_UPPER) * R_ALIGN_UPPER
                            r_low  = R.from_quat(np.array([q_l[1], q_l[2], q_l[3], q_l[0]]) * Q_MAP_LOWER) * R_ALIGN_LOWER
                            
                            # In die synchronen Optimizer schieben
                            opt_1d.add_packet_and_optimize(r_up, r_low)
                            opt_2d.add_packet_and_optimize(r_base, r_up)
                            
                print(f"✅ tau_b = {tau_b}, tau_delta = {tau_d} fertiggestellt.\n")
            log_1d = os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_1D.csv")
            log_2d = os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_2D.csv")
            
            opt_1d = SyncOptimizer1D(
                sensor_upper=ID_UPPER, sensor_lower=ID_LOWER, 
                window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
                flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
                enable_singularity_filter=ENABLE_SINGULARITY, enable_flat_valley_filter=ENABLE_FLAT_VALLEY, enable_anti_windup=ENABLE_ANTI_WINDUP, tau_b_=tau_b, tau_delta_=tau_d, delta_delta_weight=DELTA_DELTA_WEIGHT,
                log_file=log_1d, debug_log_file=None # Debug Log aus, da es das Batching extrem verlangsamt
            )
            
            opt_2d = SyncOptimizer2D(
                sensor_parent=ID_BASE, sensor_child=ID_UPPER, 
                window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
                flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD, delta_delta_weight=OPT_DELTA_DELTA_WEIGHT,
                enable_singularity_filter=ENABLE_SINGULARITY, enable_flat_valley_filter=ENABLE_FLAT_VALLEY, enable_anti_windup=ENABLE_ANTI_WINDUP, enable_limrom=ENABLE_LIMROM_2D, tau_b_=tau_b, tau_delta_=tau_d, 
                log_file=log_2d, debug_log_file=None
            )
            
            print(f"🚀 Starte Verarbeitung für tau_b = {tau_b}, tau_delta = {tau_d}...")
            
            # Unterdrücke den massiven Terminal-Spam der Optimizer während der Batch-Verarbeitung
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    for idx, row in df_raw.iterrows():
                        # Quaternionen auslesen (Fallback auf Identity, falls fehlerhaft)
                        q_b = [row.get(f'{ID_BASE}_w', 1), row.get(f'{ID_BASE}_x', 0), row.get(f'{ID_BASE}_y', 0), row.get(f'{ID_BASE}_z', 0)]
                        q_u = [row.get(f'{ID_UPPER}_w', 1), row.get(f'{ID_UPPER}_x', 0), row.get(f'{ID_UPPER}_y', 0), row.get(f'{ID_UPPER}_z', 0)]
                        q_l = [row.get(f'{ID_LOWER}_w', 1), row.get(f'{ID_LOWER}_x', 0), row.get(f'{ID_LOWER}_y', 0), row.get(f'{ID_LOWER}_z', 0)]
                        
                        # SciPy Format [x,y,z,w] & Alignment
                        r_base = R.from_quat(np.array([q_b[1], q_b[2], q_b[3], q_b[0]]) * Q_MAP_BASE) * R_ALIGN_BASE
                        r_up   = R.from_quat(np.array([q_u[1], q_u[2], q_u[3], q_u[0]]) * Q_MAP_UPPER) * R_ALIGN_UPPER
                        r_low  = R.from_quat(np.array([q_l[1], q_l[2], q_l[3], q_l[0]]) * Q_MAP_LOWER) * R_ALIGN_LOWER
                        
                        # In die synchronen Optimizer schieben
                        opt_1d.add_packet_and_optimize(r_up, r_low)
                        opt_2d.add_packet_and_optimize(r_base, r_up)
                        
            print(f"✅ tau_b = {tau_b}, tau_delta = {tau_d} fertiggestellt und gespeichert.\n")

    # --- 4. PLOTTING ---
    print("📊 Erstelle Vergleichs-Diagramm...")
    
    fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Einfluss der Zeitkonstanten ($\\tau_b$, $\\tau_\\delta$) auf den Filter", fontsize=16, fontweight='bold')
    
    # Farbpalette für alle Kombinationen generieren
    combinations = [(tb, td) for td in TAU_VALUES_TO_TEST for tb in TAU_VALUES_TO_TEST]
    colors = plt.cm.tab20(np.linspace(0, 1, len(combinations)))
    
    # Lade den ersten Durchlauf für die rohe Referenzkurve
    first_1d = pd.read_csv(os.path.join(OUTPUT_DIR, f"tau_b_{TAU_VALUES_TO_TEST[0]}_tau_d_{TAU_VALUES_TO_TEST[0]}_1D.csv"))
    t_base_1d = first_1d['time'] - first_1d['time'].iloc[0]
    axs[0].plot(t_base_1d, first_1d['delta_w'], color='lightgray', linestyle='--', linewidth=1.5, label=r"Roh-Ziel ($\Delta_w$)")
    
    first_2d = pd.read_csv(os.path.join(OUTPUT_DIR, f"tau_b_{TAU_VALUES_TO_TEST[0]}_tau_d_{TAU_VALUES_TO_TEST[0]}_2D.csv"))
    t_base_2d = first_2d['time'] - first_2d['time'].iloc[0]
    axs[2].plot(t_base_2d, first_2d['delta_w'], color='lightgray', linestyle='--', linewidth=1.5, label=r"Roh-Ziel ($\Delta_w$)")
    
    # Plotte nun die gefilterten Linien für alle Kombinationen
    for i, (tau_b, tau_d) in enumerate(combinations):
        df_1d = pd.read_csv(os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_1D.csv"))
        t_norm = df_1d['time'] - df_1d['time'].iloc[0]
        label = f"$\\tau_b$={tau_b}, $\\tau_\\delta$={tau_d}"
        
        axs[0].plot(t_norm, df_1d['delta_f_w'], color=colors[i], label=label, linewidth=1.5)
        axs[1].plot(t_norm, df_1d['b_w'], color=colors[i], linewidth=1.5)
        
        df_2d = pd.read_csv(os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_2D.csv"))
        t_norm_2 = df_2d['time'] - df_2d['time'].iloc[0]
        axs[2].plot(t_norm_2, df_2d['delta_f_w'], color=colors[i], linewidth=1.5)
        axs[3].plot(t_norm_2, df_2d['b_w'], color=colors[i], linewidth=1.5)
        
    axs[0].set_title(r"1D Gelenk (Ellenbogen) - Filtered Heading Offset ($\Delta_{f,w}$)", fontsize=12)
    axs[0].set_ylabel("Offset [Grad]")
    axs[0].grid(True)
    axs[0].legend(loc="upper left", fontsize='x-small', ncol=4)
    
    axs[1].set_title("1D Gelenk (Ellenbogen) - Berechnete Drift-Rate ($b_w$)", fontsize=12)
    axs[1].set_ylabel("Drift [Grad/Fenster]")
    axs[1].grid(True)

    axs[2].set_title(r"2D Gelenk (Schulter) - Filtered Heading Offset ($\Delta_{f,w}$)", fontsize=12)
    axs[2].set_ylabel("Offset [Grad]")
    axs[2].grid(True)

    axs[3].set_title("2D Gelenk (Schulter) - Berechnete Drift-Rate ($b_w$)", fontsize=12)
    axs[3].set_ylabel("Drift [Grad/Fenster]")
    axs[3].set_xlabel("Zeit [Sekunden]")
    axs[3].grid(True)
    
    plt.tight_layout()
    
    # Speichern und Anzeigen
    out_img = os.path.join(OUTPUT_DIR, "tau_combinations_comparison.png")
    # Lösche bestehende PNGs im Ordner
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            os.remove(os.path.join(OUTPUT_DIR, f))
            print(f"🗑️ Alte PNG gelöscht: {f}")
    plt.savefig(out_img, dpi=150)
    print(f"🎉 Fertig! Das Diagramm wurde gespeichert unter: {out_img}")
    plt.show()

    # ==============================================================================
    # --- 5. ERWEITERTE ANALYSE & RANGLISTEN ---
    # ==============================================================================
    print("\n📊 Sammle erweiterte Metriken für alle Tau-Kombinationen...")

    metrics_data = []
    combinations = [(tb, td) for td in TAU_VALUES_TO_TEST for tb in TAU_VALUES_TO_TEST]
    for tau_b, tau_d in combinations:
        log_1d = os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_1D.csv")
        log_2d = os.path.join(OUTPUT_DIR, f"tau_b_{tau_b}_tau_d_{tau_d}_2D.csv")
        
        # 1D Metriken
        df_1d = pd.read_csv(log_1d)
        active_mask_1d = ~df_1d.get('is_singular', pd.Series(False)).astype(bool)
        
        mae_1d = np.mean(np.abs(df_1d.loc[active_mask_1d, 'delta_w'] - df_1d.loc[active_mask_1d, 'delta_f_w']))
        bias_std_1d = np.std(df_1d.loc[active_mask_1d, 'b_w'])
        delta_w_var_1d = np.var(df_1d.loc[active_mask_1d, 'delta_w'])
        rms_delta_f_1d = np.sqrt(np.mean(df_1d.loc[active_mask_1d, 'delta_f_w']**2))
        num_active_1d = active_mask_1d.sum()
        sign_changes_1d = np.sum(np.diff(np.sign(df_1d.loc[active_mask_1d, 'b_w'])) != 0) if len(df_1d.loc[active_mask_1d]) > 1 else 0
        delta_f_var_1d = np.var(df_1d.loc[active_mask_1d, 'delta_f_w'])
        sign_changes_delta_f_1d = np.sum(np.diff(np.sign(df_1d.loc[active_mask_1d, 'delta_f_w'])) != 0) if len(df_1d.loc[active_mask_1d]) > 1 else 0
        
        # 2D Metriken
        df_2d = pd.read_csv(log_2d)
        active_mask_2d = ~df_2d.get('is_singular', pd.Series(False)).astype(bool)
        
        mae_2d = np.mean(np.abs(df_2d.loc[active_mask_2d, 'delta_w'] - df_2d.loc[active_mask_2d, 'delta_f_w']))
        bias_std_2d = np.std(df_2d.loc[active_mask_2d, 'b_w'])
        delta_w_var_2d = np.var(df_2d.loc[active_mask_2d, 'delta_w'])
        rms_delta_f_2d = np.sqrt(np.mean(df_2d.loc[active_mask_2d, 'delta_f_w']**2))
        num_active_2d = active_mask_2d.sum()
        sign_changes_2d = np.sum(np.diff(np.sign(df_2d.loc[active_mask_2d, 'b_w'])) != 0) if len(df_2d.loc[active_mask_2d]) > 1 else 0
        delta_f_var_2d = np.var(df_2d.loc[active_mask_2d, 'delta_f_w'])
        sign_changes_delta_f_2d = np.sum(np.diff(np.sign(df_2d.loc[active_mask_2d, 'delta_f_w'])) != 0) if len(df_2d.loc[active_mask_2d]) > 1 else 0
        
        metrics_data.append({
            'config': f"tau_b_{tau_b}_tau_d_{tau_d}",
            'tau_b': tau_b, 'tau_d': tau_d,
            'mae_1d': mae_1d, 'mae_2d': mae_2d,
            'bias_std_1d': bias_std_1d, 'bias_std_2d': bias_std_2d,
            'delta_w_var_1d': delta_w_var_1d, 'delta_w_var_2d': delta_w_var_2d,
            'rms_delta_f_1d': rms_delta_f_1d, 'rms_delta_f_2d': rms_delta_f_2d,
            'num_active_1d': num_active_1d, 'num_active_2d': num_active_2d,
            'sign_changes_1d': sign_changes_1d, 'sign_changes_2d': sign_changes_2d,
            'delta_f_var_1d': delta_f_var_1d, 'delta_f_var_2d': delta_f_var_2d,
            'sign_changes_delta_f_1d': sign_changes_delta_f_1d, 'sign_changes_delta_f_2d': sign_changes_delta_f_2d,
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Erklärungen
    print("\n🔍 Erklärungen zu Metriken:")
    print("- MAE: Mittlere Abweichung vom gemessenen delta_w (relative Genauigkeit). Niedriger = besserer Fit.")
    print("- Bias σ: Stetigkeit des Bias (Schwingungen). Niedriger = stabiler (höchste Priorität für konstante b_w).")
    print("- Vorzeichenwechsel b_w: Wie oft b_w Vorzeichen wechselt. Niedriger = konstanter (besser).")
    print("- Delta_f Varianz: Streuung des gefilterten delta_f_w (Glattheit). Niedriger = glattere Kurve.")
    print("- Vorzeichenwechsel delta_f_w: Wie oft delta_f_w Vorzeichen wechselt. Niedriger = stabiler Output.")
    print("- Delta_w Varianz: Streuung der Roh-Messung. Niedriger = konsistentere Messung.")
    print("- RMS Δ_f: Energie des gefilterten Signals. Niedriger = glatter Output.")
    print("💡 Fokus: Stetigkeit von b_w und delta_f_w (σ + Vorzeichenwechsel) ist am wichtigsten.")
    
    # Ranglisten
    print("\n🏆 Ranglisten (niedriger = besser):")
    
    categories = {
        'MAE (1D)': 'mae_1d',
        'MAE (2D)': 'mae_2d',
        'Bias Stetigkeit (σ 1D)': 'bias_std_1d',
        'Bias Stetigkeit (σ 2D)': 'bias_std_2d',
        'Vorzeichenwechsel b_w (1D)': 'sign_changes_1d',
        'Vorzeichenwechsel b_w (2D)': 'sign_changes_2d',
        'Delta_f Varianz (1D)': 'delta_f_var_1d',
        'Delta_f Varianz (2D)': 'delta_f_var_2d',
        'Vorzeichenwechsel delta_f_w (1D)': 'sign_changes_delta_f_1d',
        'Vorzeichenwechsel delta_f_w (2D)': 'sign_changes_delta_f_2d',
        'Delta_w Varianz (1D)': 'delta_w_var_1d',
        'Delta_w Varianz (2D)': 'delta_w_var_2d',
        'RMS Δ_f (1D)': 'rms_delta_f_1d',
        'RMS Δ_f (2D)': 'rms_delta_f_2d',
    }

    rank_columns = []
    for cat_name, col in categories.items():
        df_metrics[f'{col}_rank'] = df_metrics[col].rank(ascending=True, method='min')
        rank_columns.append(f'{col}_rank')
        sorted_df = df_metrics.sort_values(by=col, ascending=True)
        print(f"\n{cat_name}:")
        for rank, row in enumerate(sorted_df.head(5).itertuples(), 1):
            print(f"  {rank}. {row.config} ({getattr(row, col):.3f})")

    # Durchschnittliche Platzierung über alle Kategorien
    df_metrics['avg_rank'] = df_metrics[rank_columns].mean(axis=1)
    df_metrics['sum_rank'] = df_metrics[rank_columns].sum(axis=1)
    overall_sorted = df_metrics.sort_values(by='avg_rank')

    print("\n🏅 Gesamtrangliste basierend auf durchschnittlicher Platzierung über alle Kategorien:")
    print(f"Anzahl Kategorien: {len(rank_columns)}")
    for rank, row in enumerate(overall_sorted.head(10).itertuples(), 1):
        print(f"  {rank}. {row.config} (Durchschnittliche Platzierung: {row.avg_rank:.3f}, Rangsumme: {int(row.sum_rank)})")

    # ==============================================================================
    # --- 6. TOP 5 PLOT: BIAS UND DELTA_W IM VERGLEICH ---
    # ==============================================================================
    print("\n📊 Erstelle Top 5 Vergleichs-Plot für Bias und Delta_w...")

    top_5 = overall_sorted.head(5)
    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Top 5 Tau-Kombinationen: Vergleich von Bias (b_w) und Delta_w", fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    # Gemeinsame Roh-Delta_w Referenzkurven für 1D und 2D
    first_1d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{top_5.iloc[0].config}_1D.csv"))
    t_ref_1d = first_1d['time'] - first_1d['time'].iloc[0]
    axs[0].plot(t_ref_1d, first_1d['delta_w'], color='black', linestyle='--', linewidth=2, alpha=0.7, label="Roh Δ_w Referenz")

    first_2d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{top_5.iloc[0].config}_2D.csv"))
    t_ref_2d = first_2d['time'] - first_2d['time'].iloc[0]
    axs[2].plot(t_ref_2d, first_2d['delta_w'], color='black', linestyle='--', linewidth=2, alpha=0.7, label="Roh Δ_w Referenz")

    # Berechne gemeinsamen Roh-Bias einmal für 1D und 2D (Slope der Regression)
    active_mask_1d_ref = ~first_1d.get('is_singular', pd.Series(False)).astype(bool)
    if active_mask_1d_ref.sum() > 1:
        slope_1d_ref, _, _, _, _ = linregress(first_1d.loc[active_mask_1d_ref, 'window_index'], first_1d.loc[active_mask_1d_ref, 'delta_w'])
        true_bias_1d_ref = slope_1d_ref
    else:
        true_bias_1d_ref = 0

    active_mask_2d_ref = ~first_2d.get('is_singular', pd.Series(False)).astype(bool)
    if active_mask_2d_ref.sum() > 1:
        slope_2d_ref, _, _, _, _ = linregress(first_2d.loc[active_mask_2d_ref, 'window_index'], first_2d.loc[active_mask_2d_ref, 'delta_w'])
        true_bias_2d_ref = slope_2d_ref
    else:
        true_bias_2d_ref = 0

    # Plotte Regressionsgerade von Roh-Delta_w über Zeit in Delta_w-Plots
    slope_time_1d, intercept_time_1d, _, _, _ = linregress(t_ref_1d, first_1d['delta_w'])
    axs[0].plot(t_ref_1d, slope_time_1d * t_ref_1d + intercept_time_1d, color='red', linestyle='-', linewidth=2, alpha=0.8, label="Regressionsgerade Roh-Δ_w")

    slope_time_2d, intercept_time_2d, _, _, _ = linregress(t_ref_2d, first_2d['delta_w'])
    axs[2].plot(t_ref_2d, slope_time_2d * t_ref_2d + intercept_time_2d, color='red', linestyle='-', linewidth=2, alpha=0.8, label="Regressionsgerade Roh-Δ_w")

    # Plotte gemeinsame Roh-Bias-Linien (horizontale Linie bei Slope-Wert)
    axs[1].axhline(y=true_bias_1d_ref, color='black', linestyle='--', linewidth=2, alpha=0.7, label="Geschätzter Roh-Bias")
    axs[3].axhline(y=true_bias_2d_ref, color='black', linestyle='--', linewidth=2, alpha=0.7, label="Geschätzter Roh-Bias")

    for i, row in enumerate(top_5.itertuples()):
        config = row.config
        tau_b = row.tau_b
        tau_d = row.tau_d
        
        # 1D Daten
        df_1d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{config}_1D.csv"))
        t_norm = df_1d['time'] - df_1d['time'].iloc[0]
        
        label = f"Top {i+1}: τ_b={tau_b}, τ_δ={tau_d}"
        axs[0].plot(t_norm, df_1d['delta_f_w'], color=colors[i], linestyle='-', linewidth=2, alpha=0.8, label=f"{label} Δ_f,w")
        axs[1].plot(t_norm, df_1d['b_w'], color=colors[i], linewidth=2, label=f"{label} b_w")
        
        # 2D Daten
        df_2d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{config}_2D.csv"))
        t_norm_2 = df_2d['time'] - df_2d['time'].iloc[0]
        
        axs[2].plot(t_norm_2, df_2d['delta_f_w'], color=colors[i], linestyle='-', linewidth=2, alpha=0.8, label=f"{label} Δ_f,w")
        axs[3].plot(t_norm_2, df_2d['b_w'], color=colors[i], linewidth=2, label=f"{label} b_w")


    axs[0].set_title("1D Gelenk (Ellenbogen) - Gefiltertes Δ_{f,w} mit gemeinsamer Roh-Δ_w Referenz", fontsize=14)
    axs[0].set_ylabel("Offset [Grad]")
    axs[0].grid(True)
    axs[0].legend(loc="upper right", fontsize='small')

    axs[1].set_title("1D Gelenk (Ellenbogen) - Bias (b_w) vs. geschätzter Roh-Bias", fontsize=14)
    axs[1].set_ylabel("b_w / Raw Trend [Grad/Fenster]")
    axs[1].set_ylim([-0.1, 0])
    axs[1].grid(True)
    axs[1].legend(loc="upper right", fontsize='small', ncol=1, framealpha=0.9)

    axs[2].set_title("2D Gelenk (Schulter) - Gefiltertes Δ_{f,w} mit gemeinsamer Roh-Δ_w Referenz", fontsize=14)
    axs[2].set_ylabel("Offset [Grad]")
    axs[2].grid(True)
    axs[2].legend(loc="upper right", fontsize='small', ncol=1, framealpha=0.9)

    axs[3].set_title("2D Gelenk (Schulter) - Bias (b_w) vs. Raw Trend", fontsize=14)
    axs[3].set_ylabel("b_w / Raw Trend [Grad/Fenster]")
    axs[3].set_xlabel("Zeit [Sekunden]")
    axs[3].set_ylim([-0.1, 0])
    axs[3].grid(True)
    axs[3].legend(loc="upper right", fontsize='small', ncol=1, framealpha=0.9)

    plt.tight_layout()
    
    # Füge Auflistung der verwendeten Filter hinzu
    filter_text = (f"Getestete τ_b Werte: {', '.join(map(str, TAU_VALUES_TO_TEST))}\n"
                   f"Getestete τ_δ Werte: {', '.join(map(str, TAU_VALUES_TO_TEST))}\n"
                   f"Singularity Filter: {ENABLE_SINGULARITY}\n"
                   f"Flat Valley Filter: {ENABLE_FLAT_VALLEY}\n"
                   f"Anti-Windup: {ENABLE_ANTI_WINDUP}\n"
                   f"LimRom 2D: {ENABLE_LIMROM_2D}\n"
                   f"Delta-Delta Weight: {DELTA_DELTA_WEIGHT}\n"
                   f"Window Size: {OPT_WINDOW_SIZE}")
    plt.figtext(0.1, 0.02, filter_text, fontsize=10, ha='left', va='bottom')
    
    # Speichern (lösche alte PNGs nicht erneut, da schon oben gemacht)
    top5_img = os.path.join(OUTPUT_DIR, "top5_tau_comparison.png")
    plt.savefig(top5_img, dpi=150)
    print(f"🖼️ Top 5 Plot gespeichert unter: {top5_img}")
    plt.show()

if __name__ == "__main__":
    main()
