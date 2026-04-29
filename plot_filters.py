import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# ==========================================
# KONFIGURATION DES DASHBOARDS
# Nutze diese Toggles, um festzulegen, was geplottet werden soll.
# So bleibt das Dashboard übersichtlich!
# ==========================================
PLOT_WINDOW_RATING = False       # r_w: Bewegungsintensität (Flat Valley Erkennung)
PLOT_SINGULARITY_EVENTS = False  # is_singular: Wann griff der Notfall-Filter ein?
PLOT_HEADING_CORRECTION = True   # delta_w vs delta_f_w: Rohes vs. gefiltertes Offset
PLOT_DRIFT_RATE = False          # b_w: Die berechnete konstante Drift-Rate
PLOT_COST_FUNCTION = True        # result.fun: Wie gut konnte SciPy minimieren? (Fehlerwert)

LOG_DIR = "logs"
LOG_FILE_1D = os.path.join(LOG_DIR, "drift_log_1D.csv")
LOG_FILE_2D = os.path.join(LOG_DIR, "drift_log_2D.csv")
UPDATE_INTERVAL_MS = 1000  # Aktualisiere den Plot jede Sekunde

def update_plots(frame, axes, active_plots):
    try:
        df_1d = None
        df_2d = None
        
        if os.path.exists(LOG_FILE_1D) and os.path.getsize(LOG_FILE_1D) > 50:
            df_1d = pd.read_csv(LOG_FILE_1D)
            
        if os.path.exists(LOG_FILE_2D) and os.path.getsize(LOG_FILE_2D) > 50:
            df_2d = pd.read_csv(LOG_FILE_2D)
            
        # Zähle mit, welchen Subplot wir gerade befüllen
        plot_idx = 0

        # Axes Array linearisieren für einfachen Zugriff
        ax_list = np.atleast_1d(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for ax in ax_list:
            ax.clear()
            
        # -------------------------------------------------------------
        # 1: r_w (Rating / Bewegungsintensität)
        # -------------------------------------------------------------
        if PLOT_WINDOW_RATING:
            ax = ax_list[plot_idx]
            if df_1d is not None and not df_1d.empty:
                ax.plot(df_1d['time'], df_1d['r_w'], 'b-', label='1D (Elbow)')
            if df_2d is not None and not df_2d.empty:
                ax.plot(df_2d['time'], df_2d['r_w'], 'r-', label='2D (Shoulder)')
            ax.set_title("Window Rating (r_w) - Bewegungsintensität")
            ax.set_ylabel("Parameter r_w")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # -------------------------------------------------------------
        # 2: Singularity / Flat Valley Filter Events
        # -------------------------------------------------------------
        if PLOT_SINGULARITY_EVENTS:
            ax = ax_list[plot_idx]
            if df_1d is not None and not df_1d.empty:
                singular_1d = df_1d[df_1d['is_singular'] == 1]
                if not singular_1d.empty:
                    ax.scatter(singular_1d['time'], singular_1d['is_singular'], c='blue', label='1D Filtered', alpha=0.5)
            if df_2d is not None and not df_2d.empty:
                singular_2d = df_2d[df_2d['is_singular'] == 1]
                if not singular_2d.empty:
                    ax.scatter(singular_2d['time'], singular_2d['is_singular'], c='red', label='2D Filtered', alpha=0.5)
            ax.set_title("Singularity Block (1=Filter Aktiv)")
            ax.set_ylabel("Status (is_singular)")
            ax.set_ylim(-0.2, 1.2)
            ax.legend(loc="center right")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
        # -------------------------------------------------------------
        # 3: Heading Korrektur (delta_w vs. delta_f_w)
        # -------------------------------------------------------------
        if PLOT_HEADING_CORRECTION:
            ax = ax_list[plot_idx]
            if df_1d is not None and not df_1d.empty:
                ax.plot(df_1d['time'], np.degrees(df_1d['delta_w']), 'b--', alpha=0.5, label='1D Raw (Scipy)')
                ax.plot(df_1d['time'], np.degrees(df_1d['delta_f_w']), 'b-', label='1D Filtered')
            if df_2d is not None and not df_2d.empty:
                ax.plot(df_2d['time'], np.degrees(df_2d['delta_w']), 'r--', alpha=0.5, label='2D Raw (Scipy)')
                ax.plot(df_2d['time'], np.degrees(df_2d['delta_f_w']), 'r-', label='2D Filtered')
            ax.set_title("Heading Korrektur [Grad]")
            ax.set_ylabel("Offset in Grad")
            ax.legend(loc="lower left", fontsize="small")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
        # -------------------------------------------------------------
        # 4: Drift Rate (b_w)
        # -------------------------------------------------------------
        if PLOT_DRIFT_RATE:
            ax = ax_list[plot_idx]
            if df_1d is not None and not df_1d.empty:
                ax.plot(df_1d['time'], np.degrees(df_1d['b_w']), 'b-', label='1D Drift (b_w)')
            if df_2d is not None and not df_2d.empty:
                ax.plot(df_2d['time'], np.degrees(df_2d['b_w']), 'r-', label='2D Drift (b_w)')
            ax.set_title("Aktuelle Drift Rate [Grad / Zyklus]")
            ax.set_ylabel("Drift [Grad]")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # -------------------------------------------------------------
        # 5: Cost Function / SciPy Error
        # -------------------------------------------------------------
        if PLOT_COST_FUNCTION:
            ax = ax_list[plot_idx]
            # Achte darauf, dass 'cost_val' existiert (wurde gerade erst in optimizer.py ergänzt!)
            if df_1d is not None and not df_1d.empty and 'cost_val' in df_1d.columns:
                ax.plot(df_1d['time'], df_1d['cost_val'], 'b-', label='1D Error (Cost)')
            if df_2d is not None and not df_2d.empty and 'cost_val' in df_2d.columns:
                ax.plot(df_2d['time'], df_2d['cost_val'], 'r-', label='2D Error (Cost)')
            ax.set_title("Kostenfunktion (SciPy Fehlerwert am Minimum)")
            ax.set_ylabel("Verbleibender Fehler")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Wenn X-Achsen extrem eng sind, drehe Beschriftung
        for ax in ax_list[:plot_idx]:
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
    except Exception as e:
        print(f"Lese-Fehler beim Aktualisieren des Plots: {e}")

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        print(f"Ordner {LOG_DIR} existiert noch nicht. Bitte starte zuerst main.py / test_bridge.py")
        
    active_plots = sum([PLOT_WINDOW_RATING, PLOT_SINGULARITY_EVENTS, PLOT_HEADING_CORRECTION, PLOT_DRIFT_RATE, PLOT_COST_FUNCTION])
    
    if active_plots == 0:
        print("Alle Plots sind deaktiviert! Bitte mindestens einen Toggle auf True setzen.")
        sys.exit()

    # Dynamisches Layout berechnen
    cols = 2 if active_plots > 1 else 1
    rows = (active_plots + 1) // 2
    
    plt.style.use('default')
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    
    # Entferne leere Subplots (wenn ungerade Anzahl bei cols=2)
    if active_plots % 2 != 0 and active_plots > 1:
        if isinstance(axes, np.ndarray):
            fig.delaxes(axes.flatten()[-1])
            
    fig.canvas.manager.set_window_title('Live Singularity Filter Data')
    fig.suptitle('GIMLI / Bilbo Live Optimizer Tracking', fontsize=16, fontweight='bold')
    
    print(f"Starte Live-Plot mit {active_plots} aktiven Graphen... (Schließen, um zu beenden).")
    
    anim = FuncAnimation(fig, update_plots, fargs=(axes, active_plots), interval=UPDATE_INTERVAL_MS, cache_frame_data=False)
    plt.show()
