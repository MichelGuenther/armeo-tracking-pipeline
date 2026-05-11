import asyncio
import time
import capture2go as c2g

class SensorManager:
    def __init__(self, sensor_ids, data_queue, target_hz=50):
        """
        sensor_ids: Liste mit exakt zwei IDs, z.B. ['IMU_OBERARM', 'IMU_UNTERARM']
        data_queue: multiprocessing.Queue für die Weitergabe an den Optimizer
        target_hz: Die Zielfrequenz (entweder 50 oder 200)
        """
        self.sensor_ids = sensor_ids
        self.queue = data_queue
        self.target_hz = target_hz
        self.imus = []
        
        # "Zero-Order Hold" Prinzip: Wir speichern IMMER nur den allerneusten Wert.
        # So entkoppeln wir Schwankungen der Bluetooth-Rate von deinem Tracking-Algorithmus!
        self.latest_data = {s_id: {} for s_id in sensor_ids}
        self._init_lock = asyncio.Lock()  # Verhindert, dass Windows BLE-Befehle verschluckt, weil sie parallel gesendet werden
        self._stream_start_barrier = None # Wird bei jedem Start neu initialisiert

    async def _stream_sensor(self, imu, sensor_id):
        """Behandelt den asynchronen Datenstrom für einen einzelnen Sensor."""
        try:
            # 1. Startphase stark simplifiziert, angelehnt an das Originalskript von SensorStim
            print(f"🔄 Initialisiere Sensor: {sensor_id}...", flush=True)
            
            init_success = False
            for sub_attempt in range(3): 
                try:
                    if sub_attempt > 0:
                        # User found out: Just resending init doesn't work. We MUST clean the connection first.
                        print(f"   🧹 [{sensor_id}] Räume Verbindung auf vor weiterem Versuch...", flush=True)
                        async with self._init_lock:
                            try: await asyncio.wait_for(imu.disconnect(), timeout=2.0)
                            except: pass
                            await asyncio.sleep(1.0)
                            try: await asyncio.wait_for(imu.connect(), timeout=5.0)
                            except: pass
                            await asyncio.sleep(1.0)
                            
                    print(f"   [{sensor_id}] Init-Versuch {sub_attempt+1}/3", flush=True)
                    # Der Lock verhindert parallele GATT Commands auf Linux
                    async with self._init_lock:
                        print(f"   [{sensor_id}] Lock erhalten, sende init()...", flush=True)
                        await asyncio.wait_for(imu.init(setTime=True, abortRecording=True, abortStreaming=True), timeout=8.0)
                        await asyncio.sleep(1.0) # Kurze Atempause für BlueZ nach dem Init
                        
                    print(f"   ✅ [{sensor_id}] Init erfolgreich", flush=True)
                    init_success = True
                    break
                except asyncio.TimeoutError:
                    print(f"   ⚠️ [{sensor_id}] Init-Timeout in Versuch {sub_attempt+1}", flush=True)
                except Exception as e:
                    print(f"   ⚠️ [{sensor_id}] Init-Fehler: {type(e).__name__}: {e}", flush=True)

            if not init_success:
                raise ConnectionError(f"Alle Init-Versuche für {sensor_id} fehlgeschlagen.")

            print(f"⏳ [{sensor_id}] Warte an der Start-Barriere auf alle anderen Sensoren...", flush=True)
            await self._stream_start_barrier.wait()
            
            print(f"🚀 [{sensor_id}] Startschuss: Sende CmdSetMeasurementMode...", flush=True)
            async with self._init_lock:
                await asyncio.wait_for(imu.send(c2g.pkg.CmdSetMeasurementMode(
                    timestamp=0,
                    fullFloat200HzEnabled=False,
                    fullFixedMode=c2g.pkg.SamplingMode.MODE_DISABLED,
                    fullPackedMode=c2g.pkg.SamplingMode.MODE_DISABLED,
                    quatFloatMode=c2g.pkg.SamplingMode.MODE_200HZ,
                    quatFixedMode=c2g.pkg.SamplingMode.MODE_DISABLED,
                    quatPackedMode=c2g.pkg.SamplingMode.MODE_DISABLED,
                    statusMode=1,
                    calibDataMode=c2g.pkg.CalibrationDataMode.CALIB_DATA_DISABLED,
                    processExtensionMode=c2g.pkg.ProcessExtensionMode.NO_EXTENSION,
                    syncMode=c2g.pkg.SyncMode.NO_SYNC,
                    syncId=0,
                    disableBiasEstimation=False,
                    disableMagDistRejection=False,
                    disableMagData=True,
                )), timeout=8.0)
                
                await asyncio.wait_for(imu.send(c2g.pkg.CmdStartStreaming()), timeout=8.0)
            
            print(f"📊 [{sensor_id}] Streaming gestartet, warte auf Daten...", flush=True)
            
        except Exception as e:
            msg = f"\n❌ FEHLER: {sensor_id} konnte nicht gestartet werden! {type(e).__name__} - {e}\n"
            print(msg, flush=True)
            raise ConnectionError(msg)
        
        packet_count = 0
        last_time = time.time()

        async for package in imu:
            if 'Data' in type(package).__name__:
                try:
                    parsed = package.parse()
                    if 'quat' not in parsed:
                        continue
                    
                    packet_count += 1
                    current_time = time.time()
                    elapsed = current_time - last_time
                    if elapsed >= 1.0:
                        hz = packet_count / elapsed
                        print(f"--- SENSOR {sensor_id} FREQUENZ: {hz:.1f} Hz ---")
                        packet_count = 0
                        last_time = current_time

                    # Nur den allerneusten Wert überschreiben. Absolut KEIN Rückstand/Delay mehr!
                    self.latest_data[sensor_id] = {
                        'quat': parsed['quat']   # [w, x, y, z]
                    }
                            
                except Exception as e:
                    print(f"⚠️ Parse-Fehler bei {sensor_id}: {e}")

    async def _publisher_loop(self):
        """Pumpt mit exakt der gewünschten Frequenz den konstant neuesten Sensor-Zustand in die Hardware.
        Garantiert 100% Echtzeit, selbst wenn ein Sensor mal ein Paket verschluckt (dann wird der letzte bekannte Punkt verdoppelt, genau wie bei Videospielen!)."""
        # Blockiere den Publisher, bis WIRKLICH alle Sensoren mindestens ein Paket geschickt haben
        print("⏳ Warte auf erste eintreffende Sensor-Daten, bevor System-Tick startet...")
        while True:
            all_ready = True
            for s_id in self.sensor_ids:
                if not self.latest_data[s_id]:
                    all_ready = False
                    break
            if all_ready:
                break
            await asyncio.sleep(0.1)

        print(f"\n✅ ERFOLG: Alle {len(self.sensor_ids)} Sensoren streamen stabil!")
        interval = 1.0 / self.target_hz
        print(f"⏱️ System-Tick Rate startet und wird auf exakt {self.target_hz} Hz ({interval:.3f}s Tick) blockiert")
        
        next_time = time.time() + interval
        while True:
            start_time = time.time()
            
            # 1. Den absoluten Snapshot der aktuellen Sekunde ziehen
            sync_packet = {'timestamp': start_time}
            for s_id in self.sensor_ids:
                # Wir sichern (kopieren) den exakten Snapshot
                sync_packet[s_id] = self.latest_data[s_id].copy()
                
            # 2. Synchrones Paket für die test_bridge bereitstellen
            # put_nowait ist wichtig, damit der Loop unter gar keinen Umständen blockiert!
            try:
                self.queue.put_nowait(sync_packet)
            except:
                pass
            
            # 3. Präzise warten (Windows Timer Sleep Problem umgehen via halbem Busy-Wait)
            # Wir geben dem Event-Loop mindestens einmal Rechenzeit, um die BLE-Pakete reinzulassen.
            sleep_time = next_time - time.time()
            
            if sleep_time > 0.002:
                # Erlaube echtes Sleep, um CPU-Last zu senken (aber ziehe 2ms für den Windows-Timer-Ungenauigkeit ab)
                await asyncio.sleep(sleep_time - 0.002)
            
            # Den winzigen Rest der Zeit via sleep(0) im Kreis laufen (Spin-Lock im Async-Loop)
            while time.time() < next_time:
                await asyncio.sleep(0)
                
            next_time += interval

    async def start_gathering(self, max_retries=3):
        """Verbindet die Sensoren robust und startet die parallelen Streams."""
        print(f"\n{'='*50}\n🔵 BLUETOOTH SETUP: Suche {len(self.sensor_ids)} Sensoren:\n   {', '.join(self.sensor_ids)}\n{'='*50}")
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"🔄 Verbindungsversuch {attempt}/{max_retries}...")
                
                # Wir stellen sicher, dass keine stecken gebliebenen Daten aus vorherigen Versuchen
                # den Publisher-Loop fälschlicherweise entblocken
                for s_id in self.sensor_ids:
                    self.latest_data[s_id] = None
                    
                # Wir löschen alle alten BLE-Statusmeldungen aus capture2go mit einem Trick
                old_imus = getattr(self, 'imus', [])
                if old_imus:
                    print("🧹 Räume alte Verbindungen auf bevor ein neuer Versuch startet...", flush=True)
                    for i in old_imus:
                        try: await asyncio.wait_for(i.disconnect(), timeout=2.0) 
                        except: pass
                    await asyncio.sleep(2.0)
                
                # Die eigentliche Verbindung: Nutze custom connect mit sequentiellen Verbindungen für Linux
                try:
                    self.imus = await self._connect_with_sequential_retry(self.sensor_ids)
                except asyncio.TimeoutError:
                    print("⚠️ Timeout beim Verbinden der Sensoren (35s ausgereizt). Bluetooth Stack hängt. Neustart des Connects...")
                    self.imus = []
                
                if len(self.imus) == len(self.sensor_ids):
                    print(f"🔗 BLE-Verbindung zu {len(self.imus)} Sensoren hergestellt. Führe Handshake aus...\n")
                    self._stream_start_barrier = asyncio.Barrier(len(self.sensor_ids))
                    
                    # Tasks für Sensoren UND den synchronisierten Publisher parallel spawnen  
                    tasks = [self._publisher_loop()]
                    assigned_ids = set()
                    
                    # Massiv vereinfachte Zuordnung pro Sensor *ohne Delay* (analog zum offiziellen Skript)
                    for imu in self.imus:
                        imu_str = getattr(imu, 'name', str(imu))
                        print(f"📡 Gefundenes Gerät: {imu_str}")
                        
                        real_id = next((s_id for s_id in self.sensor_ids if s_id.lower() in imu_str.lower() and s_id not in assigned_ids), None)
                        if not real_id:
                            real_id = next(s_id for s_id in self.sensor_ids if s_id not in assigned_ids)
                            
                        assigned_ids.add(real_id)
                        tasks.append(self._stream_sensor(imu, real_id))
                    
                    # 1. Wir nutzen jetzt gather statt wait() + FIRST_EXCEPTION - genau wie im offiziellen Beispiel!
                    # asyncio.gather wirft Fehler direkt hoch. ACHTUNG: Wir müssen abbrechen, falls was crasht!
                    running_tasks = [asyncio.create_task(t) for t in tasks]
                    try:
                        await asyncio.gather(*running_tasks)
                    except asyncio.CancelledError:
                        print("Stream Tasks abgebrochen.")
                        for t in running_tasks:
                            t.cancel()
                    except Exception as e:
                        for t in running_tasks:
                            t.cancel()
                        raise ConnectionError(f"Stream-Fehler erkannt: {e}")
                    
                    return # Bricht die Retry-Schleife nach Erfolg ab
                else:
                    print(f"⚠️ Nur {len(self.imus)} von {len(self.sensor_ids)} verbunden. Breche ab und probiere es erneut, um Hänger zu vermeiden...")
                    # Kurze Pause für den Dongle/Bluetooth-Stack, um sich zu erholen
                    await asyncio.sleep(2.0)
                    
            except Exception as e:
                print(f"❌ Unerwarteter Fehler in Versuch {attempt}: {e}")
                await asyncio.sleep(2.0)
                
        raise ConnectionError("\n❌ FEHLER: Es konnten nach mehreren Versuchen nicht alle Sensoren verbunden werden. Bitte aus/einschalten!\n")

    async def _connect_with_sequential_retry(self, names):
        """
        Custom connect logic optimized for Linux/BlueZ. 
        Sequentiell verbinden statt parallel (asyncio.gather) verhindert Bluetooth-Stack-Überlastung.
        """
        print("🔍 Starte Sensor-Suche...")
        devices = {}
        for name in names:
            if name.startswith('IMU_'):
                devices[name] = None
        
        # Scan für Geräte mit Timeout
        from capture2go.ble import BleScanner
        scanner = BleScanner()
        scan_timeout = 45.0
        scan_start = time.time()
        
        try:
            async for found in scanner.scan():
                devices.update(found)
                missing = [name for name in names if devices[name] is None]
                print(f'Devices: {found}, missing: {", ".join(missing) if missing else "none"}.')
                if names and not missing:
                    print('✅ Alle Geräte gefunden, starte sequentielles Verbinden...')
                    break
                
                # Timeout für die ganze Scan-Phase
                if time.time() - scan_start > scan_timeout:
                    print(f'⚠️ Scan-Timeout nach {scan_timeout}s. Nutze gefundene Geräte.')
                    break
        except Exception as e:
            print(f"❌ Fehler beim Scannen: {e}")
            raise
        
        # Sequentiell (nicht parallel!) verbinden für Linux-Stabilität
        deviceList = [device for name in names if (device := devices[name]) is not None]
        
        if len(deviceList) != len(names):
            missing = [name for name in names if devices[name] is None]
            raise RuntimeError(f"Nicht alle Geräte gefunden: {missing}")
        
        print("🧹 Führe präventives, globales Aufräumen (Disconnect) für ALLE gefundenen Sensoren aus...", flush=True)
        for idx, imu in enumerate(deviceList, 1):
            try:
                print(f"   [{idx}/{len(deviceList)}] Trenne alte Verbindung (falls vorhanden) zu {imu.name}...", flush=True)
                await asyncio.wait_for(imu.disconnect(), timeout=3.0)
            except Exception:
                pass
        
        print("   Warte 2 Sekunden, damit der Bluetooth-Stack zur Ruhe kommt...", flush=True)
        await asyncio.sleep(2.0)
        
        print(f"🔗 Verbinde {len(deviceList)} Sensoren sequentiell...")
        for idx, imu in enumerate(deviceList, 1):
            try:
                print(f"   [{idx}/{len(deviceList)}] Verbinde {imu.name}...", flush=True)
                await asyncio.wait_for(imu.connect(), timeout=10.0)
                print(f"   ✅ {imu.name} verbunden", flush=True)
                await asyncio.sleep(1.0)  # Großzügigere Atempause für BlueZ
            except asyncio.TimeoutError:
                print(f"   ❌ Timeout bei {imu.name} nach 10s", flush=True)
                raise
            except Exception as e:
                print(f"   ❌ Fehler bei {imu.name}: {type(e).__name__}: {e}", flush=True)
                raise
        
        print("✅ Alle Sensoren erfolgreich verbunden.")
        return deviceList

# --- Test-Block: Führe diese Datei direkt aus, um zu prüfen, ob die Sensoren Daten senden ---
if __name__ == "__main__":
    from multiprocessing import Queue
    
    test_queue = Queue()
    # TRAGE HIER DEINE ECHTEN SENSOR-IDS EIN!
    manager = SensorManager(['IMU_6dee46', 'IMU_c22f23'], test_queue) # IMU_9e15c6 , IMU_c22f23 , IMU_6dee46
    
    async def test_run():
        # Starte den Manager im Hintergrund
        gather_task = asyncio.create_task(manager.start_gathering())
        
        try:
            while True:
                if not test_queue.empty():
                    packet = test_queue.get()
                    print(f"[{packet['timestamp']:.3f}] Gepacktes Frame empfangen! "
                          f"Acc1_z: {packet[manager.sensor_ids[0]]['acc'][2]:.2f} | "
                          f"Acc2_z: {packet[manager.sensor_ids[1]]['acc'][2]:.2f}")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(test_run())
    except KeyboardInterrupt:
        print("\n🛑 Datenerfassung beendet.")