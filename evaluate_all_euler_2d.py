import os
import csv
import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

# Konfiguration
CSV_FILE = "logs/raw_sensor_recording_2.csv"
PARENT_COLS = ["IMU_9e15c6_x", "IMU_9e15c6_y", "IMU_9e15c6_z", "IMU_9e15c6_w"]
CHILD_COLS  = ["IMU_6dee46_x", "IMU_6dee46_y", "IMU_6dee46_z", "IMU_6dee46_w"]

WINDOW_SEC = 2.0
STEP_SEC = 0.25
HZ = 200

WINDOW_SIZE = int(HZ * WINDOW_SEC)
STEP_SIZE = int(HZ * STEP_SEC)

METHODS = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'paper']

def residuals_euler(delta_yaw_array, r_parent_inv, r_child_window, seq):
    delta_yaw = delta_yaw_array[0]
    rot_offset = R.from_euler('z', delta_yaw, degrees=False)
    r_child_corrected = rot_offset * r_child_window
    r_rel = r_parent_inv * r_child_corrected
    
    euler_angles = r_rel.as_euler(seq, degrees=False)
    idx_x = seq.index('x')
    idx_y = seq.index('y')
    angle_x = euler_angles[:, idx_x]
    angle_y = euler_angles[:, idx_y]
    
    q_scipy = r_rel.as_quat()
    x, y, z, w = q_scipy[:, 0], q_scipy[:, 1], q_scipy[:, 2], q_scipy[:, 3]
    inner_term = np.clip(2 * w * z + 2 * x * y, -1.0, 1.0)
    angle_z = np.arcsin(inner_term)
    
    x_upper_bound = np.deg2rad(120.0)
    x_lower_bound = np.deg2rad(-60.0)
    y_upper_bound = np.deg2rad(160.0)
    y_lower_bound = np.deg2rad(-15.0)
    
    penalty_over_x = np.maximum(0, angle_x - x_upper_bound)
    penalty_under_x = np.maximum(0, x_lower_bound - angle_x)
    penalty_over_y = np.maximum(0, angle_y - y_upper_bound)
    penalty_under_y = np.maximum(0, y_lower_bound - angle_y)
    
    return np.concatenate([angle_z, penalty_over_x, penalty_under_x, penalty_over_y, penalty_under_y])

def residuals_paper(delta_yaw_array, r_parent_inv, r_child_window):
    delta_yaw = delta_yaw_array[0]
    rot_offset = R.from_euler('z', delta_yaw, degrees=False)
    r_child_corrected = rot_offset * r_child_window
    r_rel = r_parent_inv * r_child_corrected
    
    q_scipy = r_rel.as_quat()
    x, y, z, w = q_scipy[:, 0], q_scipy[:, 1], q_scipy[:, 2], q_scipy[:, 3]
    
    angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
    angle_z = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
    
    x_upper_bound = np.deg2rad(120.0)
    x_lower_bound = np.deg2rad(-60.0)
    y_upper_bound = np.deg2rad(160.0)
    y_lower_bound = np.deg2rad(-15.0)
    
    penalty_over_x = np.maximum(0, angle_x - x_upper_bound)
    penalty_under_x = np.maximum(0, x_lower_bound - angle_x)
    penalty_over_y = np.maximum(0, angle_y - y_upper_bound)
    penalty_under_y = np.maximum(0, y_lower_bound - angle_y)
    
    return np.concatenate([angle_z, penalty_over_x, penalty_under_x, penalty_over_y, penalty_under_y])

def evaluate_method(method, q_parent_full, q_child_full, timestamps_full):
    print(f"Start Evaluation für Methode: {method}")
    
    out_file = f"logs/euler_eval_{method}.csv"
    with open(out_file, "w") as f:
        f.write("time,w_idx,delta_w_deg,angle_x_deg,angle_y_deg,angle_z_deg\n")
        
    delta_f_w_minus_1 = 0.0
    
    # Fenster iterieren
    w_idx = 0
    start_idx = 0
    
    while start_idx + WINDOW_SIZE <= len(q_parent_full):
        end_idx = start_idx + WINDOW_SIZE
        
        q_parent_win = q_parent_full[start_idx:end_idx]
        q_child_win = q_child_full[start_idx:end_idx]
        t_win = timestamps_full[end_idx-1]
        
        r_parent = R.from_quat(q_parent_win)
        r_child = R.from_quat(q_child_win)
        r_parent_inv = r_parent.inv()
        
        # Initiale Grid Search nur am Anfang (wie in 1D diskutiert, sicherheitshalber auch hier)
        if w_idx == 0:
            best_initial_yaw = delta_f_w_minus_1
            lowest_coarse_cost = float('inf')
            for test_deg in range(-180, 180, 30):
                test_rad = np.deg2rad(test_deg)
                if method == 'paper':
                    res_coarse = residuals_paper([test_rad], r_parent_inv, r_child)
                else:
                    res_coarse = residuals_euler([test_rad], r_parent_inv, r_child, method)
                cost_coarse = np.sum(res_coarse**2)
                if cost_coarse < lowest_coarse_cost:
                    lowest_coarse_cost = cost_coarse
                    best_initial_yaw = test_rad
            initial_guess = [best_initial_yaw]
            delta_f_w_minus_1 = best_initial_yaw
        else:
            initial_guess = [delta_f_w_minus_1]
            
        # TRF Bounds ("Gartenzaun")
        search_radius = np.deg2rad(90.0)
        lower_bound = delta_f_w_minus_1 - search_radius
        upper_bound = delta_f_w_minus_1 + search_radius
        
        if method == 'paper':
            func = residuals_paper
            args = (r_parent_inv, r_child)
        else:
            func = residuals_euler
            args = (r_parent_inv, r_child, method)
            
        try:
            res = least_squares(
                func, 
                initial_guess, 
                args=args, 
                method='lm'
            )
            delta_w = res.x[0]
            delta_f_w_minus_1 = delta_w # Simpler Tracker (kein Filter)
        except Exception as e:
            delta_w = delta_f_w_minus_1
            
        # Final Angles berechnen
        rot_offset_best = R.from_euler('z', delta_w, degrees=False)
        r_rel_best = r_parent_inv * (rot_offset_best * r_child)
        q_best = r_rel_best.as_quat()
        x_b, y_b, z_b, w_b = q_best[:, 0], q_best[:, 1], q_best[:, 2], q_best[:, 3]
        
        if method == 'paper':
            angle_x_b = np.degrees(np.arctan2(2 * (w_b * x_b + y_b * z_b), 1 - 2 * (x_b**2 + y_b**2)))
            angle_y_b = np.degrees(np.arcsin(np.clip(2 * w_b * y_b - 2 * z_b * x_b, -1.0, 1.0)))
            angle_z_b = np.degrees(np.arcsin(np.clip(2 * w_b * z_b + 2 * x_b * y_b, -1.0, 1.0)))
        else:
            euler_b = r_rel_best.as_euler(method, degrees=True)
            angle_x_b = euler_b[:, method.index('x')]
            angle_y_b = euler_b[:, method.index('y')]
            angle_z_b = np.degrees(np.arcsin(np.clip(2 * w_b * z_b + 2 * x_b * y_b, -1.0, 1.0)))
            
        with open(out_file, "a") as f:
            f.write(f"{t_win},{w_idx},{np.degrees(delta_w):.4f},{np.mean(angle_x_b):.4f},{np.mean(angle_y_b):.4f},{np.mean(angle_z_b):.4f}\n")
            
        w_idx += 1
        start_idx += STEP_SIZE

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"❌ Datei {CSV_FILE} nicht gefunden! Prüfe den Dateinamen.")
        exit(1)
        
    df = pd.read_csv(CSV_FILE)
    q_parent_full = df[PARENT_COLS].to_numpy(dtype=float)
    q_child_full = df[CHILD_COLS].to_numpy(dtype=float)
    timestamps_full = df['timestamp'].to_numpy(dtype=float)
    
    for m in METHODS:
        evaluate_method(m, q_parent_full, q_child_full, timestamps_full)
    
    print("✅ Alle Evaluierungen abgeschlossen!")
