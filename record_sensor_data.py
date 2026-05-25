import os
import sys
import time
import csv
import asyncio
import multiprocessing
from multiprocessing import Process, Queue

# --- Imports aus deinem Projekt ---
from sensor_manager import SensorManager

# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================

# Sensor Bluetooth IDs (Wie in test_bridge.py):
ID_BASE = 'IMU_9e15c6'   # Torso/Shoulder Base
ID_UPPER = 'IMU_6dee46'  # Upper arm
ID_LOWER = 'IMU_c22f23'  # Forearm

ACTIVE_SENSORS = [ID_BASE, ID_UPPER, ID_LOWER]
SENSOR_HZ = 200

# Zieldatei
LOG_DIR = "logs"
OUTPUT_FILE = os.path.join(LOG_DIR, "raw_sensor_recording_only2Dhorizontal.csv")

# ==============================================================================

def csv_writer_process(queue, output_file, active_sensors):
    """
    Hintergrund-Prozess: Nimmt Pakete aus der Queue und schreibt sie blockierungsfrei in eine CSV.
    """
    print(f"💾 Starte Datenaufzeichnung nach: {output_file}...")
    
    # Bereite die Spalten-Köpfe (Header) vor
    header = ['timestamp']
    for s_id in active_sensors:
        header.extend([f'{s_id}_w', f'{s_id}_x', f'{s_id}_y', f'{s_id}_z'])
        
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        packets_recorded = 0
        start_time = None
        
        while True:
            try:
                # Blockiert, bis ein neues Frame vom SensorManager da ist
                packet = queue.get() 
                
                if start_time is None:
                    start_time = time.time()
                    
                # Extrahiere Daten in eine flache Liste für die CSV
                row = [packet['timestamp']]
                for s_id in active_sensors:
                    # Fallback auf Einheitsquaternion, falls ein Sensor kein 'quat' gesendet hat
                    q = packet.get(s_id, {}).get('quat', [1.0, 0.0, 0.0, 0.0])
                    row.extend(q)
                    
                writer.writerow(row)
                
                # Ausgabe-Feedback für den Nutzer
                packets_recorded += 1
                if packets_recorded % (SENSOR_HZ * 2) == 0:  # Alle 2 Sekunden Ausgabe
                    elapsed = time.time() - start_time
                    print(f"⏱️ [{elapsed:.1f}s] {packets_recorded} Pakete sicher aufgezeichnet...")
                    f.flush() # Erzwinge das Schreiben auf die Festplatte
                    
            except Exception as e:
                print(f"⚠️ Fehler beim Schreiben in CSV: {e}")
                break

async def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    data_queue = Queue()
    
    # Starte den Writer-Prozess (Daemon = Beendet sich automatisch, wenn das Hauptskript stirbt)
    writer_p = Process(target=csv_writer_process, args=(data_queue, OUTPUT_FILE, ACTIVE_SENSORS))
    writer_p.daemon = True
    writer_p.start()

    # Initialisiere deinen bestehenden stabilen Sensor Manager
    manager = SensorManager(sensor_ids=ACTIVE_SENSORS, data_queue=data_queue, target_hz=SENSOR_HZ)
    await manager.start_gathering()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Aufzeichnung durch Nutzer (Strg+C) beendet. Datei ist gespeichert!")