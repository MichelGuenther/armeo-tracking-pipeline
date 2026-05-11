import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================

OUTPUT_DIR = "logs"
RAW_CSV_FILE = "raw_sensor_data.csv"  # Beispiel, anpassen

# Beispiel-Konfigurationen (anpassen basierend auf deinem Setup)
CONFIGS_TO_TEST = {
    "Config1": {"singularity": True, "flat_valley": True, "anti_windup": True, "limrom_2d": False, "dd_weight": 0.0},
    # Füge mehr hinzu...
}

# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================

def load_metrics_from_csvs(output_dir, configs):
    """Lädt Metriken aus CSVs für alle Konfigurationen."""
    metrics_data = []
    for config_name, cfg in configs.items():
        safe_name = "".join([c if c.isalnum() else "_" for c in config_name])
        log_1d = os.path.join(output_dir, f"{safe_name}_1D.csv")
        log_2d = os.path.join(output_dir, f"{safe_name}_2D.csv")
        
        if not os.path.exists(log_1d) or not os.path.exists(log_2d):
            print(f"Warnung: CSVs für {config_name} fehlen. Überspringe.")
            continue
        
        # 1D Metriken
        df_1d = pd.read_csv(log_1d)
        active_mask_1d = ~df_1d.get('is_singular', pd.Series(False)).astype(bool)
        
        mae_1d = np.mean(np.abs(df_1d.loc[active_mask_1d, 'delta_w'] - df_1d.loc[active_mask_1d, 'delta_f_w']))
        bias_std_1d = np.std(df_1d.loc[active_mask_1d, 'b_w'])
        delta_w_var_1d = np.var(df_1d.loc[active_mask_1d, 'delta_w'])  # Varianz von delta_w als Referenz
        
        # 2D Metriken
        df_2d = pd.read_csv(log_2d)
        active_mask_2d = ~df_2d.get('is_singular', pd.Series(False)).astype(bool)
        
        mae_2d = np.mean(np.abs(df_2d.loc[active_mask_2d, 'delta_w'] - df_2d.loc[active_mask_2d, 'delta_f_w']))
        bias_std_2d = np.std(df_2d.loc[active_mask_2d, 'b_w'])
        delta_w_var_2d = np.var(df_2d.loc[active_mask_2d, 'delta_w'])
        
        metrics_data.append({
            'config': config_name,
            'mae_1d': mae_1d, 'mae_2d': mae_2d,
            'bias_std_1d': bias_std_1d, 'bias_std_2d': bias_std_2d,
            'delta_w_var_1d': delta_w_var_1d, 'delta_w_var_2d': delta_w_var_2d,
        })
    
    return pd.DataFrame(metrics_data)

# ==============================================================================
# --- RANGLISTEN-SYSTEM ---
# ==============================================================================

def compute_rankings(df_metrics):
    """Berechnet Ranglisten basierend auf Metriken."""
    print("\n🔍 Erklärungen zu Metriken (angepasst, da delta_w keine Ground Truth ist):")
    print("- MAE: Mittlere Abweichung vom gemessenen delta_w (als relative Genauigkeit). Niedriger = besserer Fit an Messung.")
    print("- Bias σ: Stetigkeit des Bias (Schwingungen um Mittelwert). Niedriger = stabiler.")
    print("- Delta_w Varianz: Streuung der Roh-Messung (Referenz für Rauschen). Niedriger = konsistentere Messung.")
    print("- Hinweis: Da delta_w keine Ground Truth ist, fokussiert die Bewertung auf relative Stabilität und Fit.")
    
    # Ranglisten
    print("\n🏆 Ranglisten (niedriger = besser):")
    
    categories = {
        'MAE (1D)': 'mae_1d',
        'MAE (2D)': 'mae_2d',
        'Bias Stetigkeit (σ 1D)': 'bias_std_1d',
        'Bias Stetigkeit (σ 2D)': 'bias_std_2d',
        'Delta_w Varianz (1D)': 'delta_w_var_1d',
        'Delta_w Varianz (2D)': 'delta_w_var_2d',
    }
    
    for cat_name, col in categories.items():
        sorted_df = df_metrics.sort_values(by=col)
        print(f"\n{cat_name}:")
        for rank, row in enumerate(sorted_df.head(5).itertuples(), 1):
            print(f"  {rank}. {row.config} ({getattr(row, col):.3f})")
    
    # Gesamtrangliste: Gewichtete Summe (MAE 40%, σ 30%, Delta_w Var 30%)
    df_metrics['score'] = (
        0.4 * (df_metrics['mae_1d'] + df_metrics['mae_2d']) / 2 +
        0.3 * (df_metrics['bias_std_1d'] + df_metrics['bias_std_2d']) / 2 +
        0.3 * (df_metrics['delta_w_var_1d'] + df_metrics['delta_w_var_2d']) / 2
    )
    overall_sorted = df_metrics.sort_values(by='score')
    print("\n🏅 Gesamtrangliste (gewichtete Metriken: MAE 40%, σ 30%, Delta_w Var 30%):")
    for rank, row in enumerate(overall_sorted.head(10).itertuples(), 1):
        print(f"  {rank}. {row.config} (Score: {row.score:.3f})")

# ==============================================================================
# --- HAUPTSKRIPT ---
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not any(f.endswith('.csv') for f in os.listdir(OUTPUT_DIR)):
        print("❌ Keine CSVs gefunden. Führe zuerst batch_evaluate_features.py aus.")
        sys.exit(1)
    
    df_metrics = load_metrics_from_csvs(OUTPUT_DIR, CONFIGS_TO_TEST)
    if df_metrics.empty:
        print("❌ Keine gültigen Metriken geladen.")
        return
    
    compute_rankings(df_metrics)

if __name__ == "__main__":
    main()