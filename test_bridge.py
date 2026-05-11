import signal
import sys
import os
import asyncio
import multiprocessing
from multiprocessing import Process, Queue
import time
import math
import numpy as np


# --- Imports ---
from sensor_manager import SensorManager
from optimizer import Optimizer1D, Optimizer2D_Universal

import queue as std_queue

# ==============================================================================
# --- ALGORITHM & SENSOR CONFIGURATION ---
# ==============================================================================

# TRACKING MODE:
# 'ALL' (Shoulder + Elbow, 50 Hz)
# '1D'  (Elbow only, 200 Hz)
# '2D'  (Shoulder only, 200 Hz)
TRACKING_MODE = 'ALL'

# Sensor Bluetooth IDs:
ID_BASE = 'IMU_9e15c6'   # Torso/Shoulder Base (Sensor 0)
ID_UPPER = 'IMU_6dee46'  # Upper arm         (Sensor 1)
ID_LOWER = 'IMU_c22f23'  # Forearm           (Sensor 2)

if TRACKING_MODE == 'ALL':
    ACTIVE_SENSORS = [ID_BASE, ID_UPPER, ID_LOWER]
    SENSOR_HZ = 200
elif TRACKING_MODE == '1D':
    ACTIVE_SENSORS = [ID_UPPER, ID_LOWER]
    SENSOR_HZ = 100
elif TRACKING_MODE == '2D':
    ACTIVE_SENSORS = [ID_BASE, ID_UPPER]
    SENSOR_HZ = 100

# Window configuration for optimization
CALCULATION_INTERVAL_SEC = 0.25
DATA_WINDOW_SEC = 2

OPT_WINDOW_SIZE = int(SENSOR_HZ * DATA_WINDOW_SEC)
OPT_STEP_SIZE = int(SENSOR_HZ * CALCULATION_INTERVAL_SEC)

# ----------------- OPTIMIZER PARAMETERS -----------------
# Toggle for Flat Valley Detection:
ENABLE_FLAT_VALLEY_FILTER = False
# Threshold for flat valley:
OPT_FLAT_VALLEY_THRESHOLD = 1e-7

# Singularity filter
ENABLE_SINGULARITY_FILTER = True

# Anti-Windup filter
ENABLE_ANTI_WINDUP = True

# LimRoM for 2D Optimizer
ENABLE_LIMROM_2D = False

TAU_B = 30
TAU_DELTA = 1
print(f"Optimierungsparameter: tau_b={TAU_B}, tau_delta={TAU_DELTA}")
print(f"Filter-Einstellungen: Singularity={ENABLE_SINGULARITY_FILTER}, FlatValley={ENABLE_FLAT_VALLEY_FILTER}, AntiWindup={ENABLE_ANTI_WINDUP}, LimRom2D={ENABLE_LIMROM_2D}")

# Cost function weight for difference in delta
OPT_DELTA_DELTA_WEIGHT = 0 #OPT_WINDOW_SIZE / math.pi

# --- LOGGING CONFIGURATION ---
ENABLE_LOGGING = False
ENABLE_DEBUG_LOGGING = False
LOG_DIR = "logs/csv"
SESSION_NAME = "session_32_tau_b30_tau_delta1_windup_nodeltadelta" # Wird in den Dateinamen der Logs eingebaut
LOG_FILE_NAME_1D = f"{SESSION_NAME}_1D.csv" # 1D Overall
LOG_FILE_NAME_2D = f"{SESSION_NAME}_2D.csv"  # 2D Overall
DEBUG_LOG_FILE_1D = f"{SESSION_NAME}_debug_1D.csv"  # 1D Debug Log
DEBUG_LOG_FILE_2D = f"{SESSION_NAME}_debug_2D.csv"  # 2D Debug Log
# ------------------------------------------------------------------

# ==============================================================================

# --- 1. WINDOWS FIXES ---
def apply_fixes():
    for sig in ['SIGHUP', 'SIGALRM', 'ITIMER_REAL']:
        if not hasattr(signal, sig): setattr(signal, sig, 1)
    _orig = signal.signal
    signal.signal = lambda s, h: _orig(s, h) if s in [signal.SIGINT, signal.SIGTERM] else None
    if not hasattr(signal, 'setitimer'): signal.setitimer = lambda *a: None


# --- 2. DER VIEWER-PROZESS (Eigene isolierte Welt) ---
def viewer_process(queue):
    apply_fixes()
    curr = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(curr, 'bilbolab', 'robots', 'gimli', 'software', 'GIMLI_Software'))
    
    from extensions.babylon.src.babylon import BabylonVisualization, BabylonObject
    from scipy.spatial.transform import Rotation as R 

    class SensorBox(BabylonObject):
        def __init__(self, id):
            super().__init__(id)
            self.object_type = 'box'
            self.type = 'box'
            self.data = {'quaternion': [1, 0, 0, 0]} # Startwert: [w, x, y, z]
        def getConfig(self): return {'width': 1, 'height': 0.3, 'depth': 1.6}
        def getData(self): return self.data

    class AngleDisplay(BabylonObject):
        def __init__(self, id, position=[0,0,0]):
            super().__init__(id)
            self.object_type = 'text'
            self.type = 'text'
            self.data = {
                'text': 'Warte auf Daten...',
                'position': position,
                'fontSize': 70,
                'color': '#FFFFFF'
            }
        def getConfig(self): return {'font': 'bold 24px Arial'}
        def getData(self): return self.data

    print("🌐 3D-Viewer Prozess startet...")
    
    lib_path = os.path.join(curr, 'bilbolab', 'robots', 'gimli', 'software', 'GIMLI_Software', 'extensions', 'babylon', 'src')
    babylon = BabylonVisualization(id='mein_viewer')
    babylon.path = lib_path 
    print(f"✅ Manueller Pfad gesetzt: {lib_path}")

    babylon.init()
    babylon.start()
    
    time.sleep(1)
    if hasattr(babylon, 'server') and babylon.server:
        print(f"📡 SERVER-CHECK: Ich lausche auf http://{babylon.server.host}:{babylon.server.port}")
    else:
        print("📡 SERVER-CHECK: Server-Objekt konnte nicht gefunden werden!")

    box0 = SensorBox('sensor0') # ID_BASE
    box1 = SensorBox('sensor1') # ID_UPPER
    box2 = SensorBox('sensor2') # ID_LOWER
    babylon.addObject(box0)
    babylon.addObject(box1)
    babylon.addObject(box2)

    # Text-Objekte für die Winkelanzeige (Positionen können hier angepasst werden)
    text_shoulder = AngleDisplay('text_shoulder', position=[0, 2.5, -2])
    text_elbow = AngleDisplay('text_elbow', position=[0, 1.5, -2])
    babylon.addObject(text_shoulder)
    babylon.addObject(text_elbow)

    log_file_1d = None
    log_file_2d = None
    debug_log_file_1d = None
    debug_log_file_2d = None

    if ENABLE_LOGGING:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        log_file_1d = os.path.join(LOG_DIR, "drift_log_1D.csv")
        log_file_2d = os.path.join(LOG_DIR, "drift_log_2D.csv")
        log_file_1d = os.path.join(LOG_DIR, LOG_FILE_NAME_1D)
        log_file_2d = os.path.join(LOG_DIR, LOG_FILE_NAME_2D)
    
    if ENABLE_DEBUG_LOGGING:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        debug_log_file_1d = os.path.join(LOG_DIR, DEBUG_LOG_FILE_1D)
        debug_log_file_2d = os.path.join(LOG_DIR, DEBUG_LOG_FILE_2D)

    # Initialize Optimizers
    optimizer_elbow = Optimizer1D(
        sensor_upper=ID_UPPER, 
        sensor_lower=ID_LOWER, 
        window_size=OPT_WINDOW_SIZE, 
        step_size=OPT_STEP_SIZE,
        flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
        enable_singularity_filter=ENABLE_SINGULARITY_FILTER,
        enable_flat_valley_filter=ENABLE_FLAT_VALLEY_FILTER,
        enable_anti_windup=ENABLE_ANTI_WINDUP,
        tau_b_=TAU_B,
        tau_delta_=TAU_DELTA,
        delta_delta_weight=OPT_DELTA_DELTA_WEIGHT,
        log_file=log_file_1d,
        debug_log_file=debug_log_file_1d
    )
    
    optimizer_shoulder = Optimizer2D_Universal(
        sensor_parent=ID_BASE, 
        sensor_child=ID_UPPER, 
        window_size=OPT_WINDOW_SIZE, 
        step_size=OPT_STEP_SIZE,
        flat_valley_threshold=OPT_FLAT_VALLEY_THRESHOLD,
        enable_singularity_filter=ENABLE_SINGULARITY_FILTER,
        enable_flat_valley_filter=ENABLE_FLAT_VALLEY_FILTER,
        enable_anti_windup=ENABLE_ANTI_WINDUP,
        enable_limrom=ENABLE_LIMROM_2D,
        tau_b_= TAU_B,
        tau_delta_=TAU_DELTA,
        delta_delta_weight=OPT_DELTA_DELTA_WEIGHT,
        log_file=log_file_2d,
        debug_log_file=debug_log_file_2d
    )
    
    # --- MANUELLE SENSOR-TO-SEGMENT KALIBRIERUNG ---
    # R_ALIGN_BASE =  R.from_euler('xyz', [-90, 0, 0], degrees=True)
    # R_ALIGN_UPPER = R.from_euler('xyz', [-90, 0, 0], degrees=True)
    # R_ALIGN_LOWER = R.from_euler('xyz', [-90, 0, 180], degrees=True)
    # MIRROR_BASE  = [1, -1, -1] 
    # MIRROR_UPPER = [1, -1, -1]
    # MIRROR_LOWER = [-1, 1, -1]
        
    R_ALIGN_BASE =  R.from_euler('xyz', [-90, 0, 0], degrees=True)
    R_ALIGN_UPPER = R.from_euler('xyz', [-90, 0, 0], degrees=True)
    R_ALIGN_LOWER = R.from_euler('xyz', [-90, 0, 180], degrees=True)
    
    # [X, Y, Z, W]
    Q_MAP_BASE  = np.array([ 1, -1, -1, 1], dtype=np.float32) 
    Q_MAP_UPPER = np.array([ 1, -1, -1, 1], dtype=np.float32)
    Q_MAP_LOWER = np.array([-1,  1, -1, 1], dtype=np.float32)
    
    packet_count = 0
    last_fps_time = time.time()
    
    try:
        while True:
            last_visual_packet = None
            
            try:
                while True:
                    packet = queue.get_nowait()
                    packet_count += 1
                    
                    # q_b = packet[ID_BASE].get('quat', [1, 0, 0, 0])  if ID_BASE  in packet else [1, 0, 0, 0]
                    # q_u = packet[ID_UPPER].get('quat', [1, 0, 0, 0]) if ID_UPPER in packet else [1, 0, 0, 0]
                    # q_l = packet[ID_LOWER].get('quat', [1, 0, 0, 0]) if ID_LOWER in packet else [1, 0, 0, 0]
                    # r_base_raw = R.from_quat([q_b[1]*MIRROR_BASE[0], q_b[2]*MIRROR_BASE[1], q_b[3]*MIRROR_BASE[2], q_b[0]])
                    # r_up_raw   = R.from_quat([q_u[1]*MIRROR_UPPER[0], q_u[2]*MIRROR_UPPER[1], q_u[3]*MIRROR_UPPER[2], q_u[0]])
                    # r_low_raw  = R.from_quat([q_l[1]*MIRROR_LOWER[0], q_l[2]*MIRROR_LOWER[1], q_l[3]*MIRROR_LOWER[2], q_l[0]])

                    q_b = packet.get(ID_BASE, {}).get('quat', [1.0, 0.0, 0.0, 0.0])
                    q_u = packet.get(ID_UPPER, {}).get('quat', [1.0, 0.0, 0.0, 0.0])
                    q_l = packet.get(ID_LOWER, {}).get('quat', [1.0, 0.0, 0.0, 0.0])
                    
                    # [x, y, z, w]
                    r_base_raw = R.from_quat(np.array([q_b[1], q_b[2], q_b[3], q_b[0]]) * Q_MAP_BASE)
                    r_up_raw   = R.from_quat(np.array([q_u[1], q_u[2], q_u[3], q_u[0]]) * Q_MAP_UPPER)
                    r_low_raw  = R.from_quat(np.array([q_l[1], q_l[2], q_l[3], q_l[0]]) * Q_MAP_LOWER)
                    
                    # 2. Manuelles Alignment (Kalibrierung) applizieren
                    r_base_aligned = r_base_raw * R_ALIGN_BASE
                    r_up_aligned = r_up_raw * R_ALIGN_UPPER
                    r_low_aligned = r_low_raw * R_ALIGN_LOWER
                    
                    # 3. Aligned-Daten bedingt in die Optimizer schieben
                    current_delta_yaw_elbow = 0.0
                    angle_elbow_x = 0.0
                    if TRACKING_MODE in ['ALL', '1D']:
                        current_delta_yaw_elbow, angle_elbow_x = optimizer_elbow.add_packet_and_optimize(r_up_aligned, r_low_aligned)
                        
                    current_delta_yaw_shoulder = 0.0
                    angles_shoulder = {'x': 0.0, 'y': 0.0}
                    if TRACKING_MODE in ['ALL', '2D']:
                        current_delta_yaw_shoulder, angles_shoulder = optimizer_shoulder.add_packet_and_optimize(r_base_aligned, r_up_aligned)
                    
                    # Speichere dir dieses letzte verarbeitete Element für den 3D Viewer!
                    last_visual_packet = {
                        'r_base': r_base_aligned,
                        'r_up': r_up_aligned,
                        'r_low': r_low_aligned,
                        'dy_elbow': current_delta_yaw_elbow,
                        'dy_sho': current_delta_yaw_shoulder,
                        'angle_elbow_x': angle_elbow_x,
                        'angles_shoulder': angles_shoulder
                    }
            except std_queue.Empty:
                pass # Queue ist komplett abgearbeitet, wir können rendern!
                
            # Frequenz (Hz) Messung & Ausgabe im Terminal
            current_time = time.time()
            elapsed_fps = current_time - last_fps_time
            if elapsed_fps >= 1.0:
                hz = packet_count / elapsed_fps
                if hz > 0: 
                    ang_info = ""
                    if last_visual_packet:
                        sx = last_visual_packet['angles_shoulder'].get('x', 0.0)
                        sy = last_visual_packet['angles_shoulder'].get('y', 0.0)
                        ex = last_visual_packet['angle_elbow_x']
                        ang_info = f" | Schulter: {sx:.1f}°, {sy:.1f}° | Ellenbogen: {ex:.1f}°"
                    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ System läuft mit: {hz:.1f} Hz (Ziel: {SENSOR_HZ} Hz){ang_info}")
                packet_count = 0
                last_fps_time = current_time
                
            # --- 2. 3D VIEWER UPDATE (Nur für den aktuellsten Zustand) ---
            # So verhindern wir, dass der Browser mit 200 * 3 = 600 WebSocket-Nachrichten pro Sekunde überschwemmt wird!
            if last_visual_packet:
                r_base_aligned = last_visual_packet['r_base']
                r_up_aligned   = last_visual_packet['r_up']
                r_low_aligned  = last_visual_packet['r_low']
                current_delta_yaw_elbow = last_visual_packet['dy_elbow']
                current_delta_yaw_shoulder = last_visual_packet['dy_sho']
                angle_elbow_x = last_visual_packet['angle_elbow_x']
                angles_shoulder = last_visual_packet['angles_shoulder']
                
                # 4. KINEMATIK: Offset auf die nachfolgenden Gelenke anwenden
                r_offset_elbow = R.from_euler('z', current_delta_yaw_elbow, degrees=False)
                r_low_corrected = r_offset_elbow * r_low_aligned
                
                r_offset_shoulder = R.from_euler('z', current_delta_yaw_shoulder, degrees=False)
                r_up_corrected = r_offset_shoulder * r_up_aligned
                
                # 6. Forward Kinematics: Lokale Gelenkrotation berechnen
                r_joint_shoulder = r_base_aligned.inv() * r_up_corrected
                r_joint_elbow = r_up_aligned.inv() * r_low_corrected
                
                # Zurück in das [w, x, y, z] Format für Babylon umwandeln
                q_base_scipy = r_base_aligned.as_quat() 
                q_joint_shoulder_scipy = r_joint_shoulder.as_quat()
                q_joint_elbow_scipy = r_joint_elbow.as_quat()
                
                q_base_send = [q_base_scipy[3], q_base_scipy[0], q_base_scipy[1], q_base_scipy[2]]
                q_joint_shoulder_send = [q_joint_shoulder_scipy[3], q_joint_shoulder_scipy[0], q_joint_shoulder_scipy[1], q_joint_shoulder_scipy[2]]
                q_joint_elbow_send = [q_joint_elbow_scipy[3], q_joint_elbow_scipy[0], q_joint_elbow_scipy[1], q_joint_elbow_scipy[2]]
                
                # Die sauberen Daten an die Visualisierung schicken
                box0.update_from_data({'quaternion': q_base_send})
                box1.update_from_data({'quaternion': q_joint_shoulder_send})
                box2.update_from_data({'quaternion': q_joint_elbow_send})

                # Winkel-Anzeige aktualisieren
                shoulder_text = f"Shoulder: {angles_shoulder.get('x', 0.0):.1f}° (X), {angles_shoulder.get('y', 0.0):.1f}° (Y)"
                elbow_text = f"Elbow: {angle_elbow_x:.1f}° (X)"
                text_shoulder.update_from_data({
                    'text': shoulder_text,
                    'position': [0, 2.5, -2],
                    'fontSize': 70,
                    'color': '#FFFFFF'
                })
                text_elbow.update_from_data({
                    'text': elbow_text,
                    'position': [0, 1.5, -2],
                    'fontSize': 70,
                    'color': '#FFFFFF'
                })
            
                box0.update()
                box1.update()
                box2.update()
                text_shoulder.update()
                text_elbow.update()
            
            # Schont den Prozessor. Babylon JS (Browser) rendert typischerweise eh nur mit 60fps (ca. 0.016s)
            time.sleep(0.015)
            
    except Exception as e:
        print(f"Fehler im Viewer-Loop: {e}")
        if hasattr(babylon, 'stop'):
            babylon.stop()


# --- 3. DER BLUETOOTH-PROZESS (Hauptprozess) ---
async def main():
    data_queue = Queue()
    
    # 1. Den Viewer-Prozess im Hintergrund starten
    p = Process(target=viewer_process, args=(data_queue,))
    p.daemon = True  # WICHTIG: Beendet den Viewer-Prozess automatisch, wenn das Terminal geschlossen wird
    p.start()

    # 2. Wir nutzen jetzt den dynamischen SensorManager!
    manager = SensorManager(sensor_ids=ACTIVE_SENSORS, data_queue=data_queue, target_hz=SENSOR_HZ)
    
    try:
        # start_gathering verbindet die IMUs und schickt die synchronisierten Pakete in die Queue
        await manager.start_gathering()
        
    except Exception as e:
        print(f"❌ Fehler im Hauptprozess: {e}")
    
    finally:
        print("🧹 Säubere Prozesse und schließe Verbindungen...")
        p.terminate()
        p.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgramm durch Nutzer beendet.")