#!/usr/bin/env python3
import sys
import os
import argparse
import asyncio
import time

# --- Purer Frequenz-Test ohne 3D-Viewer und ohne Background-Threads ---

if sys.platform == 'win32':
    os.environ["BLEAK_LOGGING_LEVEL"] = "DEBUG" 
    try:
        from bleak.backends.winrt.util import assert_mta
        async def dummy_assert_mta(): pass
        import bleak.backends.winrt.util as bleak_util
        bleak_util.assert_mta = dummy_assert_mta
    except ImportError:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
gimli_software_path = os.path.join(current_dir, 'bilbolab', 'robots', 'gimli', 'software', 'GIMLI_Software')
sys.path.insert(0, gimli_software_path)

import capture2go as c2g

async def calculate_frequency(imu, sensor_id):
    await imu.init(setTime=True, abortRecording=True, abortStreaming=True)
    
    print(f"[{sensor_id}] Sende CmdSetMeasurementMode (QUAT Float 50Hz)...")
    await imu.send(c2g.pkg.CmdSetMeasurementMode(
        timestamp=0,
        fullFloat200HzEnabled=False,
        fullFixedMode=c2g.pkg.SamplingMode.MODE_DISABLED,
        fullPackedMode=c2g.pkg.SamplingMode.MODE_DISABLED,
        quatFloatMode=c2g.pkg.SamplingMode.MODE_50HZ,   # Wir drosseln die Hardware auf saubere 50 Hz
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
    ))
    await imu.send(c2g.pkg.CmdStartStreaming())
    
    print(f"[{sensor_id}] Messung gestartet. Berechne...")
    
    packet_count = 0
    last_time = time.time()
    
    async for package in imu:
        # Wir zaehlen alle Datenpakete
        if 'Data' not in type(package).__name__:
            continue
            
        packet_count += 1
        current_time = time.time()
        elapsed = current_time - last_time
        
        if elapsed >= 1.0:
            hz = packet_count / elapsed
            print(f"--- SENSOR {sensor_id} FREQUENZ: {hz:.2f} Hz ---")
            packet_count = 0
            last_time = current_time

async def main():
    parser = argparse.ArgumentParser(description='IMU Frequenz-Test')
    parser.add_argument('devices', metavar='DEVICE', nargs='*', help='IMU device names')
    args = parser.parse_args()

    # Standard-Sensoren nutzen, falls keine per Kommandozeile übergeben wurden
    if not args.devices:
        args.devices = ['IMU_9e15c6', 'IMU_6dee46', 'IMU_c22f23']

    print(f"Versuche Verbindung zu {len(args.devices)} Sensoren aufzubauen: {args.devices}...")

    try:
        imus = await c2g.connect(args.devices)
    except Exception as e:
        return
        
    if not imus:
        return
        
    tasks = []
    for i, imu in enumerate(imus):
        tasks.append(calculate_frequency(imu, args.devices[i]))
        
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await asyncio.gather(*[imu.disconnect() for imu in imus])

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAbbruch.")

