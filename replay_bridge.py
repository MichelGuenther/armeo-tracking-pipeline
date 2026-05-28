import os
import csv
import time
import asyncio
import multiprocessing
from multiprocessing import Process, Queue

# Importiere den Viewer und die Konfiguration direkt aus deiner Live-Bridge!
from test_bridge import viewer_process, ACTIVE_SENSORS, SENSOR_HZ

# ==============================================================================
# --- KONFIGURATION FÜR DAS REPLAY ---
# ==============================================================================

# Die Datei, die wir mit record_sensor_data.py aufgenommen haben:
CSV_FILE = "logs/raw_sensor_recording_test.csv"

# Wiedergabegeschwindigkeit (1.0 = Echtzeit, 2.0 = Doppelt so schnell).
# Tipp: Setze es auf 0.0, um die Daten in Sekundenschnelle durch den 
# Optimizer zu jagen (perfekt zum schnellen Evaluieren der Logs!)
SPEED_FACTOR = 1.0 

# ==============================================================================

async def replay_csv(queue, filename):
    print(f"🎬 Starte Offline-Replay der Datei: {filename}")
    if not os.path.exists(filename):
        print(f"❌ Datei nicht gefunden: {filename}. Bitte nimm zuerst Daten auf!")
        return
        
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        
        last_csv_time = None
        packet_count = 0
        
        freq_last_time = time.time()
        freq_packet_count = 0
        
        for row in reader:
            try:
                csv_time = float(row['timestamp'])
                
                if last_csv_time is not None:
                    # Simuliere reale Zeit (Playback-Geschwindigkeit)
                    elapsed_csv = csv_time - last_csv_time
                    if elapsed_csv > 0 and SPEED_FACTOR > 0.0:
                        await asyncio.sleep(elapsed_csv / SPEED_FACTOR)
                        
                last_csv_time = csv_time
                
                # Dynamisches Mapping der Sensor-IDs (Falls sich die MAC-Adressen zwischen Replay und aktuellem Skript geändert haben)
                if packet_count == 0:
                    csv_sensors = [col.replace('_w', '') for col in row.keys() if col.endswith('_w')]
                    print(f"📡 Dynamisches Mapping für Replay: CSV Sensoren {csv_sensors} auf Active {ACTIVE_SENSORS} gemappt.")
                    # Speichere das Mapping {AktuellerSensorName: CSV_Name}
                    sensor_map = {}
                    for i, s_id in enumerate(ACTIVE_SENSORS):
                        if i < len(csv_sensors):
                            sensor_map[s_id] = csv_sensors[i]
                        else:
                            sensor_map[s_id] = None

                # Datenpaket im exakt gleichen Format wie vom SensorManager bauen
                packet = {'timestamp': csv_time}
                for s_id in ACTIVE_SENSORS:
                    csv_id = sensor_map.get(s_id)
                    if csv_id is not None:
                        try:
                            w = float(row[f'{csv_id}_w'])
                            x = float(row[f'{csv_id}_x'])
                            y = float(row[f'{csv_id}_y'])
                            z = float(row[f'{csv_id}_z'])
                            packet[s_id] = {'quat': [w, x, y, z]}
                        except KeyError:
                            packet[s_id] = {'quat': [1.0, 0.0, 0.0, 0.0]}
                    else:
                        packet[s_id] = {'quat': [1.0, 0.0, 0.0, 0.0]}
                    
                queue.put(packet)
                packet_count += 1
                freq_packet_count += 1
                
                # Berechne und printe echte Abspiel-Frequenz (Hz)
                curr_time = time.time()
                elapsed_real = curr_time - freq_last_time
                if elapsed_real >= 1.0:
                    hz = freq_packet_count / elapsed_real
                    print(f"▶️ Replay läuft... \033[93m{hz:.1f} Hz\033[0m ({packet_count} Frames total)")
                    freq_packet_count = 0
                    freq_last_time = curr_time
                    
            except Exception as e:
                print(f"⚠️ Fehler beim Lesen der CSV-Zeile: {e}")
                break
                
    print("✅ Replay beendet. Alle Daten wurden erfolgreich an den Optimizer geschickt.")

async def main():
    data_queue = Queue()
    
    # 1. Den echten Viewer-Prozess inkl. Optimizer im Hintergrund starten
    p = Process(target=viewer_process, args=(data_queue,))
    p.daemon = True
    p.start()

    # 2. Statt der echten Sensoren (SensorManager) spielen wir die CSV ab
    try:
        await replay_csv(data_queue, CSV_FILE)
        
        print("⏳ Das Replay ist fertig. Der 3D Viewer bleibt noch 10 Sekunden geöffnet...")
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Replay durch Nutzer abgebrochen.")
    finally:
        print("🧹 Schließe Prozesse...")
        p.terminate()
        p.join()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    asyncio.run(main())