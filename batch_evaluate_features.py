import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from scipy.spatial.transform import Rotation as R

from optimizer import Optimizer1D, Optimizer2D_Universal

# ==============================================================================
# --- 1. KONFIGURATION FÜR DIE EVALUATION ---
# ==============================================================================

# Die Raw-Sensor-Aufnahme
RAW_CSV_FILE = "logs/raw_sensor_recording_2.csv"

# Deine nun fixierten besten Parameter
BEST_TAU_B = 30.0
BEST_TAU_DELTA = 1.0

OUTPUT_DIR = "logs/feature_evaluation" 

# Parameter aus der Live-Bridge
SENSOR_HZ = 200
DATA_WINDOW_SEC = 1
CALCULATION_INTERVAL_SEC = 0.25
OPT_WINDOW_SIZE = int(SENSOR_HZ * DATA_WINDOW_SEC)
OPT_STEP_SIZE = int(SENSOR_HZ * CALCULATION_INTERVAL_SEC)
OPT_FLAT_VALLEY_THRESHOLD = 1e-7
OPT_DELTA_DELTA_WEIGHT = OPT_WINDOW_SIZE / math.pi

# --- SZENARIEN: Hier definierst du, welche Kombinationen gegeneinander antreten ---
"""
CONFIGS_TO_TEST = {
    "1. Nackt (Ohne Filter)": {
        "singularity": False, "flat_valley": False, "anti_windup": False, "limrom_2d": False, "dd_weight": 0.0
    },
    "2. + Singularity Filter": {
        "singularity": True,  "flat_valley": False, "anti_windup": False, "limrom_2d": False, "dd_weight": 0.0
    },
    "3. + Flat Valley Filter": {
        "singularity": True,  "flat_valley": True,  "anti_windup": False, "limrom_2d": False, "dd_weight": 0.0
    },
    "4. + Anti-Windup": {
        "singularity": True,  "flat_valley": True,  "anti_windup": True,  "limrom_2d": False, "dd_weight": 0.0
    },
    "5. Volle Pipeline (inkl. LimRom 2D & Delta-Delta)": {
        "singularity": True,  "flat_valley": True,  "anti_windup": True,  "limrom_2d": True,  "dd_weight": OPT_DELTA_DELTA_WEIGHT
    }
    # Du kannst hier jederzeit weitere Zeilen wie "Nur LimRom 2D" hinzufügen!
}
"""
import itertools

# ... dein restlicher Code (OPT_DELTA_DELTA_WEIGHT muss hier schon definiert sein) ...

CONFIGS_TO_TEST = {}

# Definiere die möglichen Zustände für jeden Parameter
options = {
    "singularity": [False, True],
    "flat_valley": [False, True],
    "anti_windup": [False, True],
    "limrom_2d":   [False, True],
    "dd_weight":   [0.0, OPT_DELTA_DELTA_WEIGHT]
}

# itertools.product erstellt uns automatisch alle 32 Kombinationen
keys = list(options.keys())
for i, values in enumerate(itertools.product(*options.values()), 1):
    config = dict(zip(keys, values))
    
    # Generiere einen sprechenden Namen für die Legende im Plot
    active_features = []
    if config["singularity"]: active_features.append("Sing")
    if config["flat_valley"]: active_features.append("FlatVal")
    if config["anti_windup"]: active_features.append("AntiW")
    if config["limrom_2d"]:   active_features.append("LimRom")
    if config["dd_weight"] != 0.0: active_features.append("DD")
    
    # Wenn die Liste leer ist, ist es der "Nackt"-Zustand
    name_str = " + ".join(active_features) if active_features else "Nackt (Ohne Filter)"
    
    # i:02d formatiert die Zahl zweistellig (01, 02, ... 32)
    dict_name = f"{i:02d}. {name_str}"
    
    CONFIGS_TO_TEST[dict_name] = config

# Ab hier kannst du CONFIGS_TO_TEST genau wie vorher in deiner Schleife benutzen!
# Sensor IDs
ID_BASE = 'IMU_9e15c6'
ID_UPPER = 'IMU_6dee46'
ID_LOWER = 'IMU_c22f23'

# Kalibrierung 
R_ALIGN_BASE =  R.from_euler('xyz', [-90, 0, 0], degrees=True)
R_ALIGN_UPPER = R.from_euler('xyz', [-90, 0, 0], degrees=True)
R_ALIGN_LOWER = R.from_euler('xyz', [-90, 0, 180], degrees=True)

Q_MAP_BASE  = np.array([ 1, -1, -1, 1], dtype=np.float32) 
Q_MAP_UPPER = np.array([ 1, -1, -1, 1], dtype=np.float32)
Q_MAP_LOWER = np.array([-1,  1, -1, 1], dtype=np.float32)

# ==============================================================================
# --- 2. SYNCHRONE OPTIMIZER ---
# ==============================================================================

class SyncOptimizer1D(Optimizer1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def add_packet_and_optimize(self, r_up_aligned, r_low_aligned):
        self.buffer_upper.append(r_up_aligned.as_quat())
        self.buffer_lower.append(r_low_aligned.as_quat())
        
        if len(self.buffer_upper) >= self.window_size:
            self._run_optimization_threaded(self.buffer_upper.copy(), self.buffer_lower.copy())
            keep_elements = max(0, self.window_size - self.step_size)
            self.buffer_upper = self.buffer_upper[-keep_elements:] if keep_elements > 0 else []
            self.buffer_lower = self.buffer_lower[-keep_elements:] if keep_elements > 0 else []

class SyncOptimizer2D(Optimizer2D_Universal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def add_packet_and_optimize(self, r_par_aligned, r_chi_aligned):
        self.buffer_parent.append(r_par_aligned.as_quat())
        self.buffer_child.append(r_chi_aligned.as_quat())
        
        if len(self.buffer_parent) >= self.window_size:
            self._run_optimization_threaded(self.buffer_parent.copy(), self.buffer_child.copy())
            keep_elements = max(0, self.window_size - self.step_size)
            self.buffer_parent = self.buffer_parent[-keep_elements:] if keep_elements > 0 else []
            self.buffer_child = self.buffer_child[-keep_elements:] if keep_elements > 0 else []

# ==============================================================================
# --- 3. HAUPTSKRIPT ---
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_raw = None
    reference_df_1d = None
    reference_df_2d = None
    print(f"🔍 DEBUG: Prüfe {len(CONFIGS_TO_TEST)} Konfigurationen...\n")
    print(f"🔍 DEBUG: OUTPUT_DIR = {OUTPUT_DIR}")
    
    if len(CONFIGS_TO_TEST) == 0:
        print("❌ ERROR: CONFIGS_TO_TEST ist leer! Keine Konfigurationen zum Testen vorhanden.")
        return

    # Lade Rohdaten einmalig
    if not os.path.exists(RAW_CSV_FILE):
        print(f"❌ Fehler: Konnte '{RAW_CSV_FILE}' nicht finden.")
        sys.exit(1)
    print(f"📂 Lese Sensor-Daten aus {RAW_CSV_FILE}...")
    df_raw = pd.read_csv(RAW_CSV_FILE)
    print(f"✅ {len(df_raw)} Zeilen geladen.\n")

    # === BERECHNE REFERENZ: 1D mit LimRom, 2D ohne LimRom ===
    print("📊 Berechne Referenz-Werte (1D mit LimRom, 2D ohne LimRom)...")
    ref_log_1d = os.path.join(OUTPUT_DIR, "reference_1D.csv")
    ref_log_2d = os.path.join(OUTPUT_DIR, "reference_2D.csv")
    
    if os.path.exists(ref_log_1d) and os.path.exists(ref_log_2d):
        print(f"📁 Gefunden: Referenz-CSVs in {OUTPUT_DIR}. Lade Referenzdaten.")
        reference_df_1d = pd.read_csv(ref_log_1d)
        reference_df_2d = pd.read_csv(ref_log_2d)
    else:
        print(f"📁 Keine Referenz-CSVs gefunden. Starte Referenz-Berechnungen...")
        opt_ref_1d = SyncOptimizer1D(
            sensor_upper=ID_UPPER, sensor_lower=ID_LOWER, 
            window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
            flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
            enable_singularity_filter=True, enable_flat_valley_filter=True, 
            enable_anti_windup=True, 
            tau_b_=BEST_TAU_B, tau_delta_=BEST_TAU_DELTA,
            delta_delta_weight=OPT_DELTA_DELTA_WEIGHT,
            log_file=ref_log_1d
        )
        
        opt_ref_2d = SyncOptimizer2D(
            sensor_parent=ID_BASE, sensor_child=ID_UPPER, 
            window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
            flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
            enable_singularity_filter=True, enable_flat_valley_filter=True,
            enable_anti_windup=True, enable_limrom=False,  # 2D OHNE LimRom als Referenz
            tau_b_=BEST_TAU_B, tau_delta_=BEST_TAU_DELTA,
            delta_delta_weight=OPT_DELTA_DELTA_WEIGHT,
            log_file=ref_log_2d
        )
        
        with contextlib.redirect_stdout(contextlib.StringIO()):
            for _, row in df_raw.iterrows():
                q_b = [float(row['IMU_6dee46_quat_w']), float(row['IMU_6dee46_quat_x']), float(row['IMU_6dee46_quat_y']), float(row['IMU_6dee46_quat_z'])]
                q_u = [float(row['IMU_9e15c6_quat_w']), float(row['IMU_9e15c6_quat_x']), float(row['IMU_9e15c6_quat_y']), float(row['IMU_9e15c6_quat_z'])]
                q_l = [float(row['IMU_c22f23_quat_w']), float(row['IMU_c22f23_quat_x']), float(row['IMU_c22f23_quat_y']), float(row['IMU_c22f23_quat_z'])]
                
                r_base = R.from_quat(np.array([q_b[1], q_b[2], q_b[3], q_b[0]]) * Q_MAP_BASE) * R_ALIGN_BASE
                r_up   = R.from_quat(np.array([q_u[1], q_u[2], q_u[3], q_u[0]]) * Q_MAP_UPPER) * R_ALIGN_UPPER
                r_low  = R.from_quat(np.array([q_l[1], q_l[2], q_l[3], q_l[0]]) * Q_MAP_LOWER) * R_ALIGN_LOWER
                
                opt_ref_1d.add_packet_and_optimize(r_up, r_low)
                opt_ref_2d.add_packet_and_optimize(r_base, r_up)
        
        reference_df_1d = pd.read_csv(ref_log_1d)
        reference_df_2d = pd.read_csv(ref_log_2d)

        # Sicherstellen, dass numerische Spalten korrekt konvertiert werden
        numeric_cols = ['time', 'delta_f_w', 'b_w', 'delta_w', 'r_w', 'cost_val', 'opt_duration', 'angle_x', 'k_b_w', 'k_delta_w']
        for col in numeric_cols:
            if col in reference_df_1d.columns:
                reference_df_1d[col] = pd.to_numeric(reference_df_1d[col], errors='coerce')
            if col in reference_df_2d.columns:
                reference_df_2d[col] = pd.to_numeric(reference_df_2d[col], errors='coerce')

        # Entferne Zeilen mit NaN in kritischen Spalten
        reference_df_1d = reference_df_1d.dropna(subset=['time', 'delta_f_w', 'b_w'])
        reference_df_2d = reference_df_2d.dropna(subset=['time', 'delta_f_w', 'b_w'])

        print(f"✅ Referenz-Werte berechnet.\n")

    # ==============================================================================
    # --- 3. KONFIGURATIONEN VERARBEITEN ---
    # ==============================================================================
    print("⚙️ Verarbeite alle Feature-Kombinationen...")

    metrics_data = []

    for config_name, cfg in CONFIGS_TO_TEST.items():
        safe_name = "".join([c if c.isalnum() else "_" for c in config_name])
        log_1d = os.path.join(OUTPUT_DIR, f"{safe_name}_1D.csv")
        log_2d = os.path.join(OUTPUT_DIR, f"{safe_name}_2D.csv")

        # Führe Optimizer aus, wenn CSVs nicht existieren
        if not os.path.exists(log_1d) or not os.path.exists(log_2d):
            print(f"⚙️ Berechne für '{config_name}'...")
            opt_1d = SyncOptimizer1D(
                sensor_upper=ID_UPPER, sensor_lower=ID_LOWER,
                window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
                flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
                enable_singularity_filter=cfg["singularity"],
                enable_flat_valley_filter=cfg["flat_valley"],
                enable_anti_windup=cfg["anti_windup"],
                tau_b_=BEST_TAU_B, tau_delta_=BEST_TAU_DELTA,
                delta_delta_weight=cfg["dd_weight"],
                log_file=log_1d, debug_log_file=None
            )

            opt_2d = SyncOptimizer2D(
                sensor_parent=ID_BASE, sensor_child=ID_UPPER,
                window_size=OPT_WINDOW_SIZE, step_size=OPT_STEP_SIZE,
                flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
                enable_singularity_filter=cfg["singularity"],
                enable_flat_valley_filter=cfg["flat_valley"],
                enable_anti_windup=cfg["anti_windup"], enable_limrom=cfg["limrom_2d"],
                tau_b_=BEST_TAU_B, tau_delta_=BEST_TAU_DELTA,
                delta_delta_weight=cfg["dd_weight"],
                log_file=log_2d, debug_log_file=None
            )

            with contextlib.redirect_stdout(io.StringIO()):
                for _, row in df_raw.iterrows():
                    q_b = [float(row['IMU_6dee46_quat_w']), float(row['IMU_6dee46_quat_x']), float(row['IMU_6dee46_quat_y']), float(row['IMU_6dee46_quat_z'])]
                    q_u = [float(row['IMU_9e15c6_quat_w']), float(row['IMU_9e15c6_quat_x']), float(row['IMU_9e15c6_quat_y']), float(row['IMU_9e15c6_quat_z'])]
                    q_l = [float(row['IMU_c22f23_quat_w']), float(row['IMU_c22f23_quat_x']), float(row['IMU_c22f23_quat_y']), float(row['IMU_c22f23_quat_z'])]

                    r_base = R.from_quat(np.array([q_b[1], q_b[2], q_b[3], q_b[0]]) * Q_MAP_BASE) * R_ALIGN_BASE
                    r_up   = R.from_quat(np.array([q_u[1], q_u[2], q_u[3], q_u[0]]) * Q_MAP_UPPER) * R_ALIGN_UPPER
                    r_low  = R.from_quat(np.array([q_l[1], q_l[2], q_l[3], q_l[0]]) * Q_MAP_LOWER) * R_ALIGN_LOWER

                    opt_1d.add_packet_and_optimize(r_up, r_low)
                    opt_2d.add_packet_and_optimize(r_base, r_up)

            print(f"✅ {config_name} fertiggestellt.")
        else:
            print(f"⏭️ Überspringe Berechnung für '{config_name}' (CSVs existieren bereits).")

        # Lade CSVs und berechne Metriken (immer)
        df_1d = pd.read_csv(log_1d)
        df_2d = pd.read_csv(log_2d)

        # Sicherstellen, dass numerische Spalten korrekt konvertiert werden
        numeric_cols = ['time', 'delta_f_w', 'b_w', 'delta_w', 'r_w', 'cost_val', 'opt_duration', 'angle_x', 'k_b_w', 'k_delta_w']
        for col in numeric_cols:
            if col in df_1d.columns:
                df_1d[col] = pd.to_numeric(df_1d[col], errors='coerce')
            if col in df_2d.columns:
                df_2d[col] = pd.to_numeric(df_2d[col], errors='coerce')

        # Entferne Zeilen mit NaN in kritischen Spalten
        df_1d = df_1d.dropna(subset=['time', 'delta_f_w', 'b_w'])
        df_2d = df_2d.dropna(subset=['time', 'delta_f_w', 'b_w'])

        # 1D Metriken
        flat_valley_mask_1d = df_1d['is_singular'].astype(bool) if 'is_singular' in df_1d.columns else pd.Series(False, index=df_1d.index)
        active_mask_1d = ~flat_valley_mask_1d

        bias_mean_1d = np.mean(df_1d.loc[active_mask_1d, 'b_w'])
        bias_var_1d = np.var(df_1d.loc[active_mask_1d, 'b_w'])
        bias_std_1d = np.std(df_1d.loc[active_mask_1d, 'b_w'])
        rms_delta_f_1d = np.sqrt(np.mean(df_1d.loc[active_mask_1d, 'delta_f_w']**2))
        delta_f_var_1d = np.var(df_1d.loc[active_mask_1d, 'delta_f_w'])
        sign_changes_1d = np.sum(np.diff(np.sign(df_1d.loc[active_mask_1d, 'b_w'])) != 0) if len(df_1d.loc[active_mask_1d]) > 1 else 0
        sign_changes_delta_f_1d = np.sum(np.diff(np.sign(df_1d.loc[active_mask_1d, 'delta_f_w'])) != 0) if len(df_1d.loc[active_mask_1d]) > 1 else 0
        num_active_1d = active_mask_1d.sum()

        # RMS-Differenz zur Referenz-delta_f_w
        interp_ref_delta_f_w_1d = np.interp(
            df_1d['time'].to_numpy(dtype=float), 
            reference_df_1d['time'].to_numpy(dtype=float), 
            reference_df_1d['delta_f_w'].to_numpy(dtype=float)
        )
        rms_diff_ref_1d = np.sqrt(np.mean((df_1d.loc[active_mask_1d, 'delta_f_w'].to_numpy(dtype=float) - interp_ref_delta_f_w_1d[active_mask_1d.to_numpy()])**2))

        # 2D Metriken
        flat_valley_mask_2d = df_2d['is_singular'].astype(bool) if 'is_singular' in df_2d.columns else pd.Series(False, index=df_2d.index)
        active_mask_2d = ~flat_valley_mask_2d

        bias_mean_2d = np.mean(df_2d.loc[active_mask_2d, 'b_w'])
        bias_var_2d = np.var(df_2d.loc[active_mask_2d, 'b_w'])
        bias_std_2d = np.std(df_2d.loc[active_mask_2d, 'b_w'])
        rms_delta_f_2d = np.sqrt(np.mean(df_2d.loc[active_mask_2d, 'delta_f_w']**2))
        delta_f_var_2d = np.var(df_2d.loc[active_mask_2d, 'delta_f_w'])
        sign_changes_2d = np.sum(np.diff(np.sign(df_2d.loc[active_mask_2d, 'b_w'])) != 0) if len(df_2d.loc[active_mask_2d]) > 1 else 0
        sign_changes_delta_f_2d = np.sum(np.diff(np.sign(df_2d.loc[active_mask_2d, 'delta_f_w'])) != 0) if len(df_2d.loc[active_mask_2d]) > 1 else 0
        num_active_2d = active_mask_2d.sum()

        # RMS-Differenz zur Referenz-delta_f_w
        interp_ref_delta_f_w_2d = np.interp(
            df_2d['time'].to_numpy(dtype=float), 
            reference_df_2d['time'].to_numpy(dtype=float), 
            reference_df_2d['delta_f_w'].to_numpy(dtype=float)
        )
        rms_diff_ref_2d = np.sqrt(np.mean((df_2d.loc[active_mask_2d, 'delta_f_w'].to_numpy(dtype=float) - interp_ref_delta_f_w_2d[active_mask_2d.to_numpy()])**2))

        metrics_data.append({
            'config': config_name,
            'bias_mean_1d': bias_mean_1d, 'bias_mean_2d': bias_mean_2d,
            'bias_std_1d': bias_std_1d, 'bias_std_2d': bias_std_2d,
            'delta_f_var_1d': delta_f_var_1d, 'delta_f_var_2d': delta_f_var_2d,
            'rms_delta_f_1d': rms_delta_f_1d, 'rms_delta_f_2d': rms_delta_f_2d,
            'sign_changes_1d': sign_changes_1d, 'sign_changes_2d': sign_changes_2d,
            'sign_changes_delta_f_1d': sign_changes_delta_f_1d, 'sign_changes_delta_f_2d': sign_changes_delta_f_2d,
            'num_active_1d': num_active_1d, 'num_active_2d': num_active_2d,
            'rms_diff_ref_1d': rms_diff_ref_1d, 'rms_diff_ref_2d': rms_diff_ref_2d,
            'limrom': cfg['limrom_2d'], 'dd_weight': cfg['dd_weight']
        })

    print(f"✅ Alle {len(CONFIGS_TO_TEST)} Konfigurationen verarbeitet.\n")

    # ==============================================================================
    # --- 4. ANALYSE & PLOTTING ---
    # ==============================================================================
    print("📊 Erstelle Feature-Vergleichs-Diagramme...")

    ks_groups = [
        {"name": "KS1_Ohne_LimRom_Ohne_DD", "title": "Kostenfunktion 1: Ohne LimRom, Ohne Delta-Delta", "limrom": False, "dd": 0.0},
        {"name": "KS2_Ohne_LimRom_Mit_DD",  "title": "Kostenfunktion 2: Ohne LimRom, Mit Delta-Delta",  "limrom": False, "dd": OPT_DELTA_DELTA_WEIGHT},
        {"name": "KS3_Mit_LimRom_Ohne_DD",  "title": "Kostenfunktion 3: Mit LimRom, Ohne Delta-Delta",  "limrom": True,  "dd": 0.0},
        {"name": "KS4_Mit_LimRom_Mit_DD",   "title": "Kostenfunktion 4: Mit LimRom, Mit Delta-Delta",   "limrom": True,  "dd": OPT_DELTA_DELTA_WEIGHT},
    ]

    for ks in ks_groups:
        # Deutlich größeres Fenster (24 breit, 20 hoch), damit nichts gequetscht wird!
        fig, axs = plt.subplots(4, 1, figsize=(24, 20), sharex=True)
        fig.suptitle(f"Ablation Study - {ks['title']}\n($\\tau_b={BEST_TAU_B}$, $\\tau_\\delta={BEST_TAU_DELTA}$)", fontsize=20, fontweight='bold')

        # Filtere Konfigurationen, die in diese KS-Gruppe gehören (es sollten genau 8 sein)
        valid_configs = {k: v for k, v in CONFIGS_TO_TEST.items() if v["limrom_2d"] == ks["limrom"] and v["dd_weight"] == ks["dd"]}
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_configs)))

        for i, (config_name, cfg) in enumerate(valid_configs.items()):
            safe_name = "".join([c if c.isalnum() else "_" for c in config_name])

            # --- DATEN 1D ---
            df_1d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_1D.csv"))
            t_norm = df_1d['time'] - df_1d['time'].iloc[0]

            # Metriken berechnen 1D
            flat_valley_mask_1d = df_1d['is_singular'].astype(bool) if 'is_singular' in df_1d.columns else pd.Series(False, index=df_1d.index)
            active_mask_1d = ~flat_valley_mask_1d

            bias_std_1d = np.std(df_1d.loc[active_mask_1d, 'b_w'])
            rms_delta_f_1d = np.sqrt(np.mean(df_1d.loc[active_mask_1d, 'delta_f_w']**2))

            # RMS-Differenz zur Referenz
            interp_ref_delta_f_w_1d = np.interp(df_1d['time'], reference_df_1d['time'], reference_df_1d['delta_f_w'])
            rms_diff_ref_1d = np.sqrt(np.mean((df_1d.loc[active_mask_1d, 'delta_f_w'] - interp_ref_delta_f_w_1d[active_mask_1d])**2))

            settings_str = f"Sing:{int(cfg['singularity'])} | Flat:{int(cfg['flat_valley'])} | AntiW:{int(cfg['anti_windup'])}"
            label_1d = f"[{settings_str}] ➔ σ_b: {bias_std_1d:.3f}° | RMS Δ_f: {rms_delta_f_1d:.3f}° | RMS Diff Ref: {rms_diff_ref_1d:.3f}°"

            axs[0].plot(t_norm, df_1d['delta_f_w'], color=colors[i], label=label_1d, linewidth=2)
            axs[1].plot(t_norm, df_1d['b_w'], color=colors[i], linewidth=2)

            # --- DATEN 2D ---
            df_2d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_2D.csv"))
            t_norm_2 = df_2d['time'] - df_2d['time'].iloc[0]

            # Metriken berechnen 2D
            flat_valley_mask_2d = df_2d['is_singular'].astype(bool) if 'is_singular' in df_2d.columns else pd.Series(False, index=df_2d.index)
            active_mask_2d = ~flat_valley_mask_2d

            bias_std_2d = np.std(df_2d.loc[active_mask_2d, 'b_w'])
            rms_delta_f_2d = np.sqrt(np.mean(df_2d.loc[active_mask_2d, 'delta_f_w']**2))

            # RMS-Differenz zur Referenz
            interp_ref_delta_f_w_2d = np.interp(df_2d['time'], reference_df_2d['time'], reference_df_2d['delta_f_w'])
            rms_diff_ref_2d = np.sqrt(np.mean((df_2d.loc[active_mask_2d, 'delta_f_w'] - interp_ref_delta_f_w_2d[active_mask_2d])**2))

            label_2d = f"[{settings_str}] ➔ σ_b: {bias_std_2d:.3f}° | RMS Δ_f: {rms_delta_f_2d:.3f}° | RMS Diff Ref: {rms_diff_ref_2d:.3f}°"

            axs[2].plot(t_norm_2, df_2d['delta_f_w'], color=colors[i], label=label_2d, linewidth=2)
            axs[3].plot(t_norm_2, df_2d['b_w'], color=colors[i], linewidth=2)

        axs[0].set_title(r"1D Gelenk (Ellenbogen) - Filtered Heading Offset ($\Delta_{f,w}$)", fontsize=16)
        axs[0].set_ylabel("Offset [Grad]", fontsize=14)
        axs[0].grid(True)
        # Legende nach rechts außen setzen!
        axs[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize='large')

        axs[1].set_title("1D Gelenk (Ellenbogen) - Berechnete Drift-Rate ($b_w$)", fontsize=16)
        axs[1].set_ylabel("Drift [Grad/Fenster]", fontsize=14)
        axs[1].set_ylim([-2, 2])
        axs[1].grid(True)

        axs[2].set_title(r"2D Gelenk (Schulter) - Filtered Heading Offset ($\Delta_{f,w}$)", fontsize=16)
        axs[2].set_ylabel("Offset [Grad]", fontsize=14)
        axs[2].grid(True)
        # Legende nach rechts außen setzen!
        axs[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize='large')

        axs[3].set_title("2D Gelenk (Schulter) - Berechnete Drift-Rate ($b_w$)", fontsize=16)
        axs[3].set_ylabel("Drift [Grad/Fenster]", fontsize=14)
        axs[3].set_xlabel("Zeit [Sekunden]", fontsize=14)
        axs[3].set_ylim([-2, 2])
        axs[3].grid(True)

        # Mache Platz auf der rechten Seite, damit die Legende nicht abgeschnitten wird
        plt.tight_layout(rect=[0, 0, 0.70, 1.0])

        out_img = os.path.join(OUTPUT_DIR, f"Ablation_{ks['name']}.png")
        # bbox_inches='tight' sorgt dafür, dass die äußere Legende mitgespeichert wird
        plt.savefig(out_img, dpi=150, bbox_inches='tight')
        print(f"🖼️ Bild gespeichert: {out_img}")

    print("🎉 Alle 4 Diagramme wurden erfolgreich exportiert!")
    plt.show()

    # ==============================================================================
    # --- 5. ERWEITERTE ANALYSE & RANGLISTEN ---
    # ==============================================================================
    print("\n📊 Erstelle Ranglisten...")

    df_metrics = pd.DataFrame(metrics_data)

    # Erklärungen ausgeben
    print("\n🔍 Erklärungen zu Metriken:")
    print("- Bias Ø: Durchschnitt der berechneten Drift-Rate (b_w). Sollte nahe 0 sein, wenn kein systematischer Drift vorliegt.")
    print("- Bias σ: Standardabweichung von b_w. Niedriger = stabilere Bias-Schätzung.")
    print("- Δ_f Varianz: Streuung des gefilterten Offsets. Niedriger = glattere Filterausgabe.")
    print("- RMS Δ_f: Root Mean Square des gefilterten Offsets. Niedriger = weniger Ausreißer/Spikes.")
    print("- Vorzeichenwechsel b_w: Wie oft die Drift-Rate ihr Vorzeichen ändert. Niedriger = konsistenter Bias.")
    print("- Vorzeichenwechsel Δ_f_w: Wie oft der gefilterte Offset sein Vorzeichen wechselt. Niedriger = stabilerer Output.")
    print("- Aktive Fenster: Anzahl der nicht-singulären Auswertungen. Höher = mehr verwertbare Daten.")
    print("- RMS Diff Referenz Δ_f_w: Root Mean Square der Differenz zwischen dem gefilterten Offset und dem Referenz-Offset. Niedriger = besserer Fit zur Referenz.")
    print("- Hinweis: delta_w ist hier keine Ground Truth, sondern Teil der internen Offset-Berechnung.")

    # Ranglisten
    print("\n🏆 Ranglisten (niedriger = besser, außer bei Aktive Fenster):")

    categories = {
        'Bias Stabilität (σ 1D)': 'bias_std_1d',
        'Bias Stabilität (σ 2D)': 'bias_std_2d',
        'Vorzeichenwechsel b_w (1D)': 'sign_changes_1d',
        'Vorzeichenwechsel b_w (2D)': 'sign_changes_2d',
        'Delta_f Varianz (1D)': 'delta_f_var_1d',
        'Delta_f Varianz (2D)': 'delta_f_var_2d',
        'Vorzeichenwechsel Δ_f_w (1D)': 'sign_changes_delta_f_1d',
        'Vorzeichenwechsel Δ_f_w (2D)': 'sign_changes_delta_f_2d',
        'RMS Δ_f (1D)': 'rms_delta_f_1d',
        'RMS Δ_f (2D)': 'rms_delta_f_2d',
        'RMS Diff Referenz Δ_f_w (1D)': 'rms_diff_ref_1d',
        'RMS Diff Referenz Δ_f_w (2D)': 'rms_diff_ref_2d',
        'Aktive Fenster (1D)': 'num_active_1d',
        'Aktive Fenster (2D)': 'num_active_2d'
    }

    rank_columns = []
    for cat_name, col in categories.items():
        order = 'desc' if 'Aktive' in cat_name else 'asc'
        df_metrics[f'{col}_rank'] = df_metrics[col].rank(ascending=(order == 'asc'), method='min')
        rank_columns.append(f'{col}_rank')
        sorted_df = df_metrics.sort_values(by=col, ascending=(order == 'asc'))
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
    # --- 6. TOP 5 PLOT ---
    # ==============================================================================
    print("\n📊 Erstelle Top 5 Vergleichs-Plot...")

    top_5 = overall_sorted.head(5)
    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Top 5 Feature-Kombinationen: Vergleich von bias/stability und Δ_f_w", fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for i, row in enumerate(top_5.itertuples()):
        config = row.config
        safe_name = "".join([c if c.isalnum() else "_" for c in config])

        # 1D Daten
        df_1d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_1D.csv"))
        t_norm = df_1d['time'] - df_1d['time'].iloc[0]

        label = f"Top {i+1}: {config}"
        axs[0].plot(t_norm, df_1d['delta_f_w'], color=colors[i], linestyle='-', linewidth=2, alpha=0.8, label=f"{label} Δ_f,w")
        axs[1].plot(t_norm, df_1d['b_w'], color=colors[i], linewidth=2, label=f"{label} b_w")

        # 2D Daten
        df_2d = pd.read_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_2D.csv"))
        t_norm_2 = df_2d['time'] - df_2d['time'].iloc[0]

        axs[2].plot(t_norm_2, df_2d['delta_f_w'], color=colors[i], linestyle='-', linewidth=2, alpha=0.8, label=f"{label} Δ_f,w")
        axs[3].plot(t_norm_2, df_2d['b_w'], color=colors[i], linewidth=2, label=f"{label} b_w")

    axs[0].set_title("1D Gelenk (Ellenbogen) - Gefiltertes Δ_f,w für Top 5 Feature-Kombinationen", fontsize=14)
    axs[0].set_ylabel("Offset [Grad]")
    axs[0].grid(True)
    axs[0].legend(loc="upper right", fontsize='small')

    axs[1].set_title("1D Gelenk (Ellenbogen) - Bias (b_w) für Top 5 Feature-Kombinationen", fontsize=14)
    axs[1].set_ylabel("b_w [Grad/Fenster]")
    axs[1].grid(True)
    axs[1].legend(loc="upper right", fontsize='small', ncol=1, framealpha=0.9)

    axs[2].set_title("2D Gelenk (Schulter) - Gefiltertes Δ_f,w für Top 5 Feature-Kombinationen", fontsize=14)
    axs[2].set_ylabel("Offset [Grad]")
    axs[2].grid(True)
    axs[2].legend(loc="upper right", fontsize='small', ncol=1, framealpha=0.9)

    axs[3].set_title("2D Gelenk (Schulter) - Bias (b_w) für Top 5 Feature-Kombinationen", fontsize=14)
    axs[3].set_ylabel("b_w [Grad/Fenster]")
    axs[3].set_xlabel("Zeit [Sekunden]")
    axs[3].grid(True)
    axs[3].legend(loc="upper right", fontsize='small', ncol=1, framealpha=0.9)

    plt.tight_layout()

    # Füge Auflistung der verwendeten Filter hinzu
    filter_text = (f"Fixierte τ_b: {BEST_TAU_B}\n"
                   f"Fixierte τ_δ: {BEST_TAU_DELTA}\n"
                   f"Window Size: {OPT_WINDOW_SIZE}\n"
                   f"Step Size: {OPT_STEP_SIZE}\n"
                   f"Flat Valley Threshold: {OPT_FLAT_VALLEY_THRESHOLD}\n"
                   f"Delta-Delta Weight: {OPT_DELTA_DELTA_WEIGHT}")
    plt.figtext(0.1, 0.02, filter_text, fontsize=10, ha='left', va='bottom')

    # Speichern
    top5_img = os.path.join(OUTPUT_DIR, "top5_feature_comparison.png")
    plt.savefig(top5_img, dpi=150)
    print(f"🖼️ Top 5 Plot gespeichert unter: {top5_img}")
    plt.show()

    print("\n🎉 Feature-Evaluation abgeschlossen!")

if __name__ == "__main__":
    main()