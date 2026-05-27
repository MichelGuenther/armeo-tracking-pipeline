import asyncio
import time
import capture2go as c2g

class SensorManager:
    def __init__(self, sensor_ids, data_queue, target_hz=50):
        """
        sensor_ids: Liste mit exakt zwei/drei IDs, z.B. ['IMU_OBERARM', 'IMU_UNTERARM']
        data_queue: multiprocessing.Queue für die Weitergabe an den Optimizer
        target_hz: Die Zielfrequenz (entweder 50 oder 200)
        """
        self.sensor_ids = sensor_ids
        self.queue = data_queue
        self.target_hz = target_hz
        self.imus = []
        
        # "Zero-Order Hold" Prinzip: Wir speichern IMMER nur den allerneusten Wert.
        self.latest_data = {s_id: None for s_id in sensor_ids}

    async def _setup_and_stream(self, imu, sensor_id):
        """Konfiguriert und startet den Sensor ohne zeitaufwendiges init() und ohne Hardware-Sync."""
        try:
            print(f"🚀 [{sensor_id}] Sende CmdSetMeasurementMode (Fast Mode ohne Sync)...", flush=True)
            mode_cmd = c2g.pkg.CmdSetMeasurementMode(
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
                syncMode=c2g.pkg.SyncMode.NO_SYNC,  # <--- OHNE SYNC
                syncId=0,
                disableBiasEstimation=False,
                disableMagDistRejection=False,
                disableMagData=True,
            )
            
            # Befehl senden und auf Acknowledgment warten
            await asyncio.wait_for(imu.sendAndAwaitAck(mode_cmd, c2g.pkg.DataMeasurementMode), timeout=5.0)
            
            # WICHTIG: Puffer komplett leeren!
            async for package in imu:
                if isinstance(package, c2g.pkg.DataMeasurementMode):
                    break
                    
            # Streaming starten
            await asyncio.wait_for(imu.send(c2g.pkg.CmdStartStreaming()), timeout=5.0)
            print(f"📊 [{sensor_id}] Streaming gestartet!", flush=True)
            
        except Exception as e:
            msg = f"\n❌ FEHLER: Setup für {sensor_id} fehlgeschlagen! {e}\n"
            print(msg, flush=True)
            raise ConnectionError(msg)
            
        # --- Stream-Schleife ---
        packet_count = 0
        last_time = time.time()

        async for package in imu:
            if 'Data' in type(package).__name__:
                try:
                    parsed = package.parse()
                    if 'quat' not in parsed:
                        continue
                        
                    # Daten im "Zero-Order Hold" Speicher ablegen
                    self.latest_data[sensor_id] = parsed
                    
                    # Frequenz-Monitoring
                    packet_count += 1
                    current_time = time.time()
                    elapsed = current_time - last_time
                    if elapsed >= 1.0:
                        hz = packet_count / elapsed
                        print(f"--- SENSOR {sensor_id} FREQUENZ: \033[93m{hz:.1f} Hz\033[0m ---")
                        packet_count = 0
                        last_time = current_time

                except Exception as e:
                    print(f"Fehler beim Parsen von {sensor_id}: {e}")

    async def _publisher_loop(self):
        """Verpackt die neuesten Sensordaten synchronisiert und schickt sie mit target_hz an den Optimizer."""
        print(f"📡 Publisher Loop gestartet mit {self.target_hz} Hz.", flush=True)
        interval = 1.0 / self.target_hz
        next_time = time.time() + interval

        while True:
            # Baue das gemeinsame Paket
            packet = {'timestamp': time.time()}
            all_ready = True
            
            for s_id in self.sensor_ids:
                data = self.latest_data[s_id]
                if data is None:
                    all_ready = False
                    break
                packet[s_id] = data

            if all_ready:
                try:
                    self.queue.put_nowait(packet)
                except Exception:
                    pass

            # Präziser Sleep für den Publisher-Loop
            sleep_time = next_time - time.time()
            if sleep_time > 0.002:
                await asyncio.sleep(sleep_time - 0.002)
            
            while time.time() < next_time:
                await asyncio.sleep(0)
                
            next_time += interval

    async def start_gathering(self, max_retries=3):
        """Smarte, schnelle Parallelsuch- und Verbindungslogik ohne Timeouts."""
        print(f"\n{'='*50}\n🔵 BLUETOOTH SETUP: Suche {len(self.sensor_ids)} Sensoren (FAST MODE):\n   {', '.join(self.sensor_ids)}\n{'='*50}")
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"🔄 Verbindungsversuch {attempt}/{max_retries}...")
                
                # Resets
                for s_id in self.sensor_ids:
                    self.latest_data[s_id] = None
                
                # Paralleler Connect über capture2go! (Sehr schnell, scannt & verbindet alle auf einmal)
                try:
                    self.imus = await asyncio.wait_for(c2g.connect(self.sensor_ids), timeout=30.0)
                except asyncio.TimeoutError:
                    print("⚠️ Timeout beim Verbinden der Sensoren. Neustart des Connects...")
                    self.imus = []
                
                if self.imus and len(self.imus) == len(self.sensor_ids):
                    print(f"🔗 BLE-Verbindung zu {len(self.imus)} Sensoren hergestellt. Starte Streams...\n")
                    
                    tasks = [self._publisher_loop()]
                    assigned_ids = set()
                    
                    for imu in self.imus:
                        # Sensor ID Mapping
                        imu_str = getattr(imu, 'name', str(imu))
                        real_id = next((s_id for s_id in self.sensor_ids if s_id.lower() in imu_str.lower() and s_id not in assigned_ids), None)
                        if not real_id:
                            real_id = next(s_id for s_id in self.sensor_ids if s_id not in assigned_ids)
                        assigned_ids.add(real_id)
                        
                        # Starte Setup und Streaming GEMEINSAM in einem Task für jeden Sensor
                        tasks.append(self._setup_and_stream(imu, real_id))
                    
                    running_tasks = [asyncio.create_task(t) for t in tasks]
                    try:
                        # Lässt alle Streams und den Publisher gleichzeitig laufen
                        await asyncio.gather(*running_tasks)
                    except asyncio.CancelledError:
                        print("Stream Tasks abgebrochen.")
                        for t in running_tasks:
                            t.cancel()
                    except Exception as e:
                        for t in running_tasks:
                            t.cancel()
                        raise ConnectionError(f"Stream-Fehler erkannt: {e}")
                    
                    return # Erfolg!
                else:
                    print(f"⚠️ Nur {len(self.imus) if self.imus else 0} von {len(self.sensor_ids)} verbunden. Trenne und Retry...")
                    if self.imus:
                        # Trenne alles, bevor wir es erneut versuchen
                        await asyncio.gather(*[imu.disconnect() for imu in self.imus], return_exceptions=True)
                    await asyncio.sleep(2.0)
                    
            except Exception as e:
                print(f"❌ Unerwarteter Fehler in Versuch {attempt}: {e}")
                await asyncio.sleep(2.0)
                
        raise ConnectionError("\n❌ FEHLER: Es konnten nach mehreren Versuchen nicht alle Sensoren verbunden werden. Bitte Dongle resetten!\n")

if __name__ == "__main__":
    from multiprocessing import Queue
    test_queue = Queue()
    manager = SensorManager(['IMU_9e15c6', 'IMU_c22f23'], test_queue)
    try:
        asyncio.run(manager.start_gathering())
    except KeyboardInterrupt:
        print("\n🛑 Datenerfassung beendet.")
