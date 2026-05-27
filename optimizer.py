import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import threading
from scipy.optimize import least_squares

class Optimizer1D:
    """
    This optimizer implements the 1D joint constraints for a hinge joint (e.g., elbow).
    Since the magnetometers are disabled, the IMUs accumulate a heading drift (yaw drift) around the global Z-axis.
    Assumption: The elbow (as a hinge joint) anatomically allows only one axis of rotation (flexion/extension). 
    Movements on the other two axes (abduction/adduction and internal/external rotation) are anatomical impossibilities (or "forbidden" axes).
    The optimizer collects data in a sliding window and searches for the heading correction angle (delta_yaw) 
    that minimizes the measured movement (variance) on these two "forbidden" axes.
    """
    def __init__(self, sensor_upper, sensor_lower, window_size=200, step_size=100, flat_valley_threshold=1e-4, enable_singularity_filter=True, enable_flat_valley_filter=True, enable_anti_windup=True, enable_valley_retry_validation=True, tau_b_=2.8, tau_delta_=0.7, delta_delta_weight=0.0, limrom_mode="off", mode_kinematic_constraints=False, log_file="drift_log_1D.csv", debug_log_file=None):
        """
        Initializes the 1D optimizer for a sensor pair.
        
        Args:
            sensor_upper (str): String ID of the upper arm sensor (parent).
            sensor_lower (str): String ID of the forearm sensor (child).
            window_size (int): Number of samples for the optimization window.
            step_size (int): After how many new samples the calculation should be *repeated*.
                             If step_size = window_size, there is no overlap (tiles back-to-back).
            flat_valley_threshold: Threshold (variance) from which movement is sufficient for finding a minimum.
            debug_log_file (str, optional): If provided, saves all tested heading offsets and their cost values for debugging.
        """
        self.s_upper = sensor_upper
        self.s_lower = sensor_lower
        self.window_size = window_size
        self.step_size = step_size
        self.flat_valley_threshold = flat_valley_threshold
        self.enable_singularity_filter = enable_singularity_filter
        self.enable_flat_valley_filter = enable_flat_valley_filter
        self.enable_anti_windup = enable_anti_windup
        self.mode_kinematic_constraints = mode_kinematic_constraints
        self.limrom_mode = limrom_mode
        if self.limrom_mode == "dual_seed_referee":
            self.mode_kinematic_constraints = True
            
        self.rom_violation_counter = 0  # Hysteresis Counter
        self.delta_delta_weight = delta_delta_weight
        self.log_file = log_file
        self.debug_log_file = debug_log_file
        
        self.latest_angle_x = 0.0
        
        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write("time,window_index,r_w,is_singular,delta_w,b_w,delta_f_w,cost_val,opt_duration,angle_x,k_b_w,k_delta_w,valley_jump_occurred,seed_lost_occurred\n")
        
        if self.debug_log_file:
            with open(self.debug_log_file, "w") as f:
                f.write("time,window_index,tested_yaw_deg,cost_val,is_best,movement_var_up,movement_var_low,is_flat_valley,r_w,best_yaw_deg\n")
        
        # Buffers for the sliding window
        self.buffer_upper = []
        self.buffer_lower = []
        
        self.is_calculating = False # Prevents thread traffic jams
        
        # Target offset from the optimizer and the smoothed current offset
        self.target_heading_offset = 0.0
        self.current_heading_offset = 0.0
        
        # --- Heading Filter States (Paper Eq. 15-20) ---
        self.w_index = 0 # window index (iterations)
        self.b_w_minus_1 = 0.0 # old bias
        self.delta_w_minus_1 = 0.0 # old heading offset (delta_w)
        self.delta_f_w_minus_1 = 0.0 # old filtered heading offset (delta_f_w)
        self.seed_B_w_minus_1 = np.pi # old alternative seed for 2D tracking
        self.T_s = step_size / 200.0  # Window duration in sec 
        self.tau_b = tau_b_              # Tunable time constant for bias filter
        self.tau_delta = tau_delta_           # Tunable time constant for heading filter
        self.r_min = 0.1              # Singularity detection
        
        # --- Discontinuity Detection (Auto-Reset on Reconnection) ---
        self.last_quat_upper = None  # Track last quaternion to detect sensor reconnections
        self.last_quat_lower = None
        self.discontinuity_threshold = np.deg2rad(60.0)  # 60° jump = new connection

    def add_packet_and_optimize(self, r_up_aligned, r_low_aligned):
        """
        Takes aligned sensor rotations (SciPy) from the manager/bridge,
        fills the sliding window and triggers the asynchronous optimization once the window is full.
        """
        q_up = r_up_aligned.as_quat()
        q_low = r_low_aligned.as_quat()
        
        # --- AUTO-RESET ON SENSOR RECONNECTION ---
        # Detect discontinuities (large quaternion jumps > 60°) indicating sensor reconnection
        if self.last_quat_upper is not None:
            # Calculate angular distance between consecutive quaternions
            q_diff_up = R.from_quat(self.last_quat_upper).inv() * R.from_quat(q_up)
            q_diff_low = R.from_quat(self.last_quat_lower).inv() * R.from_quat(q_low)
            # as_rotvec() gives rotation vector; magnitude = rotation angle in radians
            angle_dist_up = np.linalg.norm(q_diff_up.as_rotvec())
            angle_dist_low = np.linalg.norm(q_diff_low.as_rotvec())
            
            if angle_dist_up > self.discontinuity_threshold or angle_dist_low > self.discontinuity_threshold:
                print(f"🔄 [1D] SENSOR RECONNECTION DETECTED! (Jump: {np.degrees(max(angle_dist_up, angle_dist_low)):.1f}°) Resetting optimizer state...")
                self._reset_filter_state()
        
        self.last_quat_upper = q_up
        self.last_quat_lower = q_low
        
        self.buffer_upper.append(q_up)
        self.buffer_lower.append(q_low)
        
        if len(self.buffer_upper) >= self.window_size:
            if not self.is_calculating:
                buf_up_copy = self.buffer_upper.copy()
                buf_low_copy = self.buffer_lower.copy()
                
                t = threading.Thread(target=self._run_optimization_threaded, args=(buf_up_copy, buf_low_copy))
                t.daemon = True
                t.start()
            # Sliding Window Mechanism: Delete step_size old elements 
            keep_elements = max(0, self.window_size - self.step_size)
            self.buffer_upper = self.buffer_upper[-keep_elements:] if keep_elements > 0 else []
            self.buffer_lower = self.buffer_lower[-keep_elements:] if keep_elements > 0 else []

        # Always return the currently found
        # target directly (no visual faking/smoothing)
        self.current_heading_offset = self.target_heading_offset
        return self.current_heading_offset, self.latest_angle_x

    def _reset_filter_state(self):
        """Reset all filter states to prevent unbounded drift on sensor reconnection."""
        self.b_w_minus_1 = 0.0
        self.delta_w_minus_1 = 0.0
        self.delta_f_w_minus_1 = 0.0
        self.seed_B_w_minus_1 = np.pi
        self.target_heading_offset = 0.0
        self.current_heading_offset = 0.0
        self.rom_violation_counter = 0
        self.buffer_upper = []
        self.buffer_lower = []
    
    def _eval_limrom_cost(self, delta_yaw, r_upper_inv, r_lower_window):
        orig_limrom = self.limrom_mode
        self.limrom_mode = "limrom_referee" # Temporär für Referee-Entscheidung aktivieren
        orig_weight = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        res = self._residuals([delta_yaw], r_upper_inv, r_lower_window)
        cost = np.sum(res**2)
        self.limrom_mode = orig_limrom
        self.delta_delta_weight = orig_weight
        return cost

    def _residuals(self, delta_yaw_array, r_upper_inv, r_lower_window):
        delta_yaw = delta_yaw_array[0]
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_lower_corrected = rot_offset * r_lower_window
        r_rel = r_upper_inv * r_lower_corrected
        
        q = r_rel.as_quat()
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # --- EULER ANGLE EXTRACTION (cf. Eq. 13) ---
        angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        angle_z = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        
        upper_bound = np.deg2rad(96.0)
        lower_bound = np.deg2rad(-60.0)
        
        # --- ANATOMICAL CONSTRAINTS (LimRoM PENALTY) ---
        # Excludes mirrored or anatomically impossible orientations (the "second valley")
        # by strictly penalizing states outside valid joint bounds (cf. Eq. 14).
        # least_squares quadriert automatisch. Für einen Faktor von 100 müssen wir hier * 10.0 nehmen.
        penalty_over = np.maximum(0, angle_x - upper_bound) * 2.0
        penalty_under = np.maximum(0, lower_bound - angle_x) * 2.0
        
        res_list = [angle_y, angle_z]#, penalty_over, penalty_under]
            
        if self.delta_delta_weight > 0.0:
            regularization_residual = (self.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.delta_delta_weight)
            res_list.append([regularization_residual])
            
        residuals = np.concatenate(res_list)
        
        # --- DEBUG LOGGING: Speichere jeden von least_squares getesteten Punkt ---
        if self.debug_log_file and hasattr(self, '_current_debug_info'):
            cost = np.sum(residuals**2)
            info = self._current_debug_info
            with open(self.debug_log_file, "a") as f:
                # is_best ist temporär 0, da wir erst am Ende das echte Minimum loggen
                f.write(f"{time.time()},{info['w_index']},{np.degrees(delta_yaw):.4f},{cost:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")
        
        # Gibt das 1D-Array aller Restfehler zurück an SciPy
        return residuals

    def _run_optimization_threaded(self, buf_up, buf_low):
        """
        Executes the Grid Search optimization algorithm in the background.
        """
        self.is_calculating = True
        try:
            # --- STATIONARY STATE DETECTION ("FLAT VALLEY") ---
            # If the sensors are completely still, the optimization problem becomes underconstrained.
            # We detect this state to freeze the pose update and rely on bias extrapolation instead.
            quat_diffs_up = np.diff(buf_up, axis=0)
            quat_diffs_low = np.diff(buf_low, axis=0)
            movement_var_up = np.sum(quat_diffs_up**2)
            movement_var_low = np.sum(quat_diffs_low**2)
            
            # Movement threshold: is it a flat valley?
            is_flat_valley = (movement_var_up < self.flat_valley_threshold and movement_var_low < self.flat_valley_threshold)
            print(f"🔍 [1D] Variance Check | Up: {movement_var_up:.6E}, Low: {movement_var_low:.6E} | Threshold: {self.flat_valley_threshold:.6E}")
            
            # 1. Convert lists into Rotation() objects (N-dimensional)
            r_upper = R.from_quat(buf_up)
            r_lower = R.from_quat(buf_low)
            
            # Massive speedup: We invert the upper arm window ONCE in advance
            r_upper_inv = r_upper.inv()
            
            # --- WINDOW RATING r_w ---
            # For the hinge joint there is only one main axis, we take local X [1, 0, 0] here.
            z_comp = r_upper.apply([1, 0, 0])[:, 2]
            r_w_k = np.sqrt(np.clip(1.0 - z_comp**2, 0.0, 1.0))
            r_w = np.sqrt(np.mean(r_w_k**2))
            
            opt_start = time.time()
            
            # Logik: Soll die Optimierung übersprungen werden?
            should_bypass_opt = is_flat_valley and self.enable_flat_valley_filter

            # --- STATIONARY PERIOD BYPASS ("FLAT VALLEY") ---
            if should_bypass_opt:
                # We skip the search entirely and freeze the old value update 
                # (bias will still extrapolate) to protect the model from noise.
                delta_w = self.delta_w_minus_1
                opt_duration = time.time() - opt_start
                optimization_success = True
                cost_fun_val = 0.0  # For the logger
                
                # Write a single debug line for flat valley indicating no search occurred
                if self.debug_log_file:
                    with open(self.debug_log_file, "a") as f:
                        f.write(f"{time.time()},{self.w_index + 1},{np.degrees(delta_w):.4f},{cost_fun_val:.6f},1,{movement_var_up:.6E},{movement_var_low:.6E},1,{r_w:.4f},{np.degrees(delta_w):.4f}\n")
            else:
                # --- LEVENBERG-MARQUARDT (GAUSS-NEWTON) OPTIMIZATION ---
                # Der perfekte Startpunkt aus dem vorherigen Fenster
                initial_guess = [self.delta_f_w_minus_1]
                # Grid Search nur im 'classic' Modus
                if self.w_index == 0 and getattr(self, 'limrom_mode', 'classic') == 'classic':
                    best_cost_init = float('inf')
                    for test_deg in range(-180, 180, 15):
                        test_rad = np.deg2rad(test_deg)
                        cost_init = np.sum(self._residuals([test_rad], r_upper_inv, r_lower)**2)
                        if cost_init < best_cost_init:
                            best_cost_init = cost_init
                            initial_guess = [test_rad]
                    print(f"🌍 [1D] Initial Grid Search abgeschlossen. Starte bei {np.degrees(initial_guess[0]):.1f}°")
                
                self._current_debug_info = {
                    'w_index': self.w_index + 1,
                    'var_up': movement_var_up,
                    'var_low': movement_var_low,
                    'is_flat': int(is_flat_valley),
                    'r_w': r_w
                }
                
                # --- COARSE SEARCH FOR PLOTTING ---
                if self.debug_log_file:
                    info = self._current_debug_info
                    del self._current_debug_info
                    for test_deg in range(-180, 180, 5):
                        test_rad = np.deg2rad(test_deg)
                        res_coarse = self._residuals([test_rad], r_upper_inv, r_lower)
                        cost_coarse = np.sum(res_coarse**2)
                        with open(self.debug_log_file, "a") as f:
                            f.write(f"{time.time()},{info['w_index']},{test_deg:.4f},{cost_coarse:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")
                    
                    # LOGGE DIE START SEEDS FÜR DAS PLOTTEN!
                    seed_a = self.delta_f_w_minus_1
                    # Korrekte Verschiebung um 180° (pi) und Wrapping in [-pi, pi]:
                    seed_b = (seed_a + np.pi + np.pi) % (2 * np.pi) - np.pi
                    
                    cost_a_seed = np.sum(self._residuals([seed_a], r_upper_inv, r_lower)**2)
                    cost_b_seed = np.sum(self._residuals([seed_b], r_upper_inv, r_lower)**2)
                    
                    # LimRom Evaluation
                    lr_cost_a = self._eval_limrom_cost(seed_a, r_upper_inv, r_lower)
                    lr_cost_b = self._eval_limrom_cost(seed_b, r_upper_inv, r_lower)
                    
                    with open(self.debug_log_file, "a") as f:
                        f.write(f"{time.time()},{info['w_index']},{np.degrees(seed_a):.4f},{cost_a_seed:.6f},3,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},{lr_cost_a:.6f}\n")
                        f.write(f"{time.time()},{info['w_index']},{np.degrees(seed_b):.4f},{cost_b_seed:.6f},4,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},{lr_cost_b:.6f}\n")
                    self._current_debug_info = info
                
                self.latest_jump_event = False
                self.latest_seed_lost = False
                
                if self.mode_kinematic_constraints:
                    # --- 1D DUAL SEED REFEREE (VEREINFACHT) ---
                    # 1. Optimierung im aktuellen Tal (gestartet beim letzten Filter-Wert)
                    res_A = least_squares(self._residuals, [self.delta_f_w_minus_1], args=(r_upper_inv, r_lower), method='lm')
                    sol_A = res_A.x[0]
                    sol_A_norm = (sol_A + np.pi) % (2 * np.pi) - np.pi
                    
                    # 2. Den exakt gespiegelten Punkt (180 Grad entfernt) als Seed B nehmen
                    sol_B_norm = (sol_A_norm + np.pi) % (2 * np.pi) - np.pi
                    
                    # Wir prüfen, ob im 180° gespiegelten Punkt ein echtes Minimum existiert, indem wir dort kurz optimieren
                    res_B = least_squares(self._residuals, [sol_B_norm], args=(r_upper_inv, r_lower), method='lm')
                    sol_B = res_B.x[0]
                    sol_B_norm = (sol_B + np.pi) % (2 * np.pi) - np.pi
                    
                    # 3. Berechne Totale Kosten (Kinematic + LimRom) für beide Täler
                    orig_limrom = self.limrom_mode
                    self.limrom_mode = "limrom_referee"
                    
                    cost_total_A = np.sum(self._residuals([sol_A_norm], r_upper_inv, r_lower)**2)
                    cost_total_B = np.sum(self._residuals([sol_B_norm], r_upper_inv, r_lower)**2)
                    
                    self.limrom_mode = orig_limrom
                    
                    # 4. Flip-Entscheidung: Ist das 180° gespiegelte Tal BESSER als unser aktuelles?
                    # Da B oft astronomisch hoch ist (weil es kein Minimum ist), passiert meist nichts.
                    # Aber in Singularitäten (flaches Tal) entscheidet LimRom, und B könnte gewinnen.
                    if cost_total_B < cost_total_A:
                        self.rom_violation_counter += 1
                        if self.rom_violation_counter >= 3:
                            delta_yaw = sol_B_norm
                            cost_fun_val = cost_total_B
                            print(f"🔀 [1D] DUAL SEED FLIP: Tal B ({cost_total_B:.1f}) hat gewonnen gegen Tal A ({cost_total_A:.1f}).")
                            self.rom_violation_counter = 0
                        else:
                            delta_yaw = sol_A_norm
                            cost_fun_val = cost_total_A
                    else:
                        self.rom_violation_counter = max(0, self.rom_violation_counter - 1)
                        delta_yaw = sol_A_norm
                        cost_fun_val = cost_total_A
                            
                    # Debug Logging for chosen Minima
                    if self.debug_log_file:
                        is_A_best = 1 if delta_yaw == sol_A_norm else 2
                        is_B_best = 1 if delta_yaw == sol_B_norm else 2
                        with open(self.debug_log_file, "a") as f:
                            f.write(f"{time.time()},{self.w_index + 1},{np.degrees(sol_A_norm):.4f},{cost_total_A:.6f},{is_A_best},{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(delta_yaw):.4f}\n")
                            f.write(f"{time.time()},{self.w_index + 1},{np.degrees(sol_B_norm):.4f},{cost_total_B:.6f},{is_B_best},{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(delta_yaw):.4f}\n")
                            
                    delta_w = delta_yaw
                    optimization_success = True
                    opt_duration = time.time() - opt_start
                else:
                    # --- LEVENBERG-MARQUARDT (STANDARD) ---
                    res = least_squares(
                        self._residuals, 
                        initial_guess, 
                        args=(r_upper_inv, r_lower), 
                        method='lm'
                    )
                    best_yaw = res.x[0]
                    best_cost = res.cost * 2.0  # res.cost ist bei scipy immer 0.5 * sum(residuals**2)
                    
                    if self.debug_log_file:
                        with open(self.debug_log_file, "a") as f:
                            # Log the starting seed
                            f.write(f"{time.time()},{self.w_index + 1},{np.degrees(initial_guess[0]):.4f},0.0,3,{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(initial_guess[0]):.4f}\n")
                            # Log the actual optimized minimum
                            f.write(f"{time.time()},{self.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                    
                    opt_duration = time.time() - opt_start
                    optimization_success = res.success 
                    delta_w = best_yaw
                    cost_fun_val = best_cost
            
            if optimization_success:
                self.w_index += 1
                w_idx = self.w_index
                
                # Retrieve the angle_x (Gelenkwinkel / joint angle) from the last buffer element
                q_up = R.from_quat(buf_up[-1]).inv()
                q_low_corrected = R.from_euler('z', delta_w, degrees=False) * R.from_quat(buf_low[-1])
                q_rel = q_up * q_low_corrected
                q = q_rel.as_quat()
                # Extracted according to eq. 13 limits
                angle_x = np.arctan2(2 * (q[3] * q[0] + q[1] * q[2]), 1 - 2 * (q[0]**2 + q[1]**2))
                self.latest_angle_x = angle_x
                
                # --- FILTER GAINS ---
                k_b_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                # --- HEADING FILTER / BIAS EXTRAPOLATION ---
                if self.enable_singularity_filter:
                    # Ein Flat Valley Event zählt nur dann als singulär, wenn der Filter dafür auch aktiviert ist
                    is_singular_due_to_flat_valley = is_flat_valley and self.enable_flat_valley_filter
                    is_singular = (r_w < self.r_min) or is_singular_due_to_flat_valley
                    s_w = 0.0 if is_singular else r_w
                    
                    if is_singular_due_to_flat_valley:
                        print(f"💤 [1D] Flat Valley ACTIVE! Arm held still. Using bias extrapolation.")
                    elif is_singular:
                        print(f"⚠️ [1D] Singularity filter ACTIVE! (Rating (r_w): {r_w:.3f} < {self.r_min}). Using bias extrapolation.")
##############################################################################
                    # --- STATE OVERRIDE (ANTI WIND-UP) ---
                    if is_singular and self.enable_anti_windup:
                        # Verhindert, dass ein fehlerhafter Müllwert für den nächsten Frame gespeichert wird
                        delta_w = self.delta_f_w_minus_1 + self.b_w_minus_1
##############################################################################
                else:
                    is_singular = False
                    s_w = 1.0 # Trust fully without dampening
                
                # --- JUMP STRATEGY / VALLEY TRACKING ---
                if self.mode_kinematic_constraints:
                    jump_diff = (delta_w - self.delta_f_w_minus_1 + np.pi) % (2 * np.pi) - np.pi
                    if np.abs(jump_diff) > np.deg2rad(120.0):
                        # Es fand ein Valley Jump statt!
                        jump_offset = jump_diff
                        self.latest_jump_event = True
                
                # Filter smoothing parameters based on time constants
                k_b_w =     max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                # --- JUMP DETECTION AND FILTER ALIGNMENT ---
                # Check for 1D Valley Jumps
                if hasattr(self, 'jump_offset') and self.jump_offset != 0.0:
                    jump_offset = self.jump_offset
                    self.delta_w_minus_1 = (self.delta_w_minus_1 + jump_offset + np.pi) % (2 * np.pi) - np.pi
                    self.delta_f_w_minus_1 = (self.delta_f_w_minus_1 + jump_offset + np.pi) % (2 * np.pi) - np.pi
                    if abs(jump_offset) > np.deg2rad(50.0):
                        print(f"🦘 [1D] Valley Jump detektiert! Filter-Historie um exakt {np.degrees(jump_offset):.1f}° verschoben.")
                        
                # Update bias (learned yaw drift rate) - Angle-Safe Version!
                if w_idx == 1:
                    b_w = 0.0
                    delta_f_w = delta_w
                else:
                    diff_w_w1 = (delta_w - self.delta_w_minus_1 + np.pi) % (2 * np.pi) - np.pi
                    b_w = self.b_w_minus_1 + s_w * k_b_w * (diff_w_w1 - self.b_w_minus_1)
                    
                    diff_w_fw1 = (delta_w - self.delta_f_w_minus_1 + np.pi) % (2 * np.pi) - np.pi
                    delta_f_w = self.delta_f_w_minus_1 + b_w + s_w * k_delta_w * (diff_w_fw1 - b_w)
                
                delta_f_w = (delta_f_w + np.pi) % (2 * np.pi) - np.pi
                
                # Remember filter state for the next window
                self.b_w_minus_1 = b_w
                self.delta_w_minus_1 = delta_w
                self.delta_f_w_minus_1 = delta_f_w
                
                self.target_heading_offset = delta_f_w
                self.current_heading_offset = self.target_heading_offset
                
                # Berechne den finalen Flexions-Winkel (angle_x) der gefundenen "besten" Lösung für das Log
                rot_offset_best = R.from_euler('z', delta_w, degrees=False)
                q_best = (r_upper_inv * (rot_offset_best * r_lower)).as_quat()
                x_b, y_b, z_b, w_b = q_best[:, 0], q_best[:, 1], q_best[:, 2], q_best[:, 3]
                best_angle_x = np.arctan2(2 * (w_b * x_b + y_b * z_b), 1 - 2 * (x_b**2 + y_b**2))
                self.latest_angle_x = np.degrees(np.mean(best_angle_x))
                avg_angle_x_deg = np.degrees(np.mean(best_angle_x))
                
                print(f"\033[95m[{time.strftime('%H:%M:%S')}]\033[0m 1D Optimizer (Elbow):")
                print(f"    -> Filter Pipeline: Raw LM: \033[96m{np.degrees(delta_w):.2f}°\033[0m | Filtered Out: \033[38;5;208m{np.degrees(delta_f_w):.2f}°\033[0m | Bias: \033[93m{np.degrees(b_w):.4f}°\033[0m")
                print(f"    -> Modus: \033[93m{getattr(self, 'limrom_mode', 'off')}\033[0m | Kinematic Cost (LM Error): \033[91m{cost_fun_val:.6f}\033[0m")
                print(f"    -> GELENKWINKEL 1D (Flexion/Extension): \033[92m{self.latest_angle_x:.1f}°\033[0m")
                print(f"    -> Visibility (\033[94mr_w: {r_w:.2f}\033[0m) | Singularity: \033[91m{is_singular}\033[0m | duration: {opt_duration:.3f}s")
                
                # Logging
                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"{time.time()},{w_idx},{r_w:.4f},{int(is_singular)},{np.degrees(delta_w):.4f},{np.degrees(b_w):.4f},{np.degrees(delta_f_w):.4f},{cost_fun_val:.6f},{opt_duration:.4f},{self.latest_angle_x:.2f},{k_b_w:.6f},{k_delta_w:.6f},{int(self.latest_jump_event)},{int(self.latest_seed_lost)}\n")
                        
        except Exception as e:
            print(f"Error in optimizer thread: {e}")
        finally:
            self.is_calculating = False

class Optimizer2D_Universal:
    """
    This optimizer implements the 2D joint constraints for a universal joint (e.g., simplified shoulder).
    A 2-DoF joint allows rotation around two axes (e.g., flexion/extension and abduction/adduction), 
    but blocks the third axis (e.g., the axial internal/external rotation along the bone).
    The optimizer searches for the heading correction angle (delta_yaw) that minimizes the variance 
    on this single "forbidden" axis.
    """
    def __init__(self, sensor_parent, sensor_child, window_size=200, step_size=100, flat_valley_threshold=1e-8, enable_singularity_filter=True, enable_flat_valley_filter=True, enable_anti_windup=True, enable_valley_retry_validation=True, enable_limrom=False, limrom_mode="off", mode_kinematic_constraints=False, tau_b_=2.8, tau_delta_=0.7, delta_delta_weight=0.0, log_file="drift_log_2D.csv", debug_log_file=None):
        self.s_parent = sensor_parent
        self.s_child = sensor_child
        self.window_size = window_size
        self.step_size = step_size
        self.flat_valley_threshold = flat_valley_threshold
        self.enable_singularity_filter = enable_singularity_filter
        self.enable_flat_valley_filter = enable_flat_valley_filter
        self.enable_anti_windup = enable_anti_windup
        self.enable_limrom = enable_limrom
        self.delta_delta_weight = delta_delta_weight
        self.log_file = log_file
        self.debug_log_file = debug_log_file

        self.limrom_mode = limrom_mode
        self.mode_kinematic_constraints = False
        if self.limrom_mode == "dual_seed_referee":
            self.mode_kinematic_constraints = True
            
        self.latest_angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.latest_jump_event = False
        self.latest_seed_lost = False

        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write("time,window_index,r_w,is_singular,delta_w,b_w,delta_f_w,cost_val,opt_duration,angle_x,angle_y,angle_z,k_b_w,k_delta_w,valley_jump_occurred,seed_lost_occurred\n")
        
        if self.debug_log_file:
            with open(self.debug_log_file, "w") as f:
                f.write("time,window_index,tested_yaw_deg,cost_val,is_best,movement_var_parent,movement_var_child,is_flat_valley,r_w,best_yaw_deg\n")
        
        self.buffer_parent = []
        self.buffer_child = []
        
        self.is_calculating = False
        self.target_heading_offset = 0.0
        self.current_heading_offset = 0.0
        
        # --- Heading Filter States (Paper Eq. 15-20) ---
        self.w_index = 0
        self.b_w_minus_1 = 0.0
        self.delta_w_minus_1 = 0.0
        self.delta_f_w_minus_1 = 0.0
        self.seed_B_w_minus_1 = np.pi
        self.T_s = step_size / 200.0  # Window duration in sec (Assumption: 100 Hz DataFrame)
        self.tau_b = tau_b_             # Tunable time constant for bias filter
        self.tau_delta = tau_delta_          # Tunable time constant for heading filter
        self.r_min = 0.1              # Empirically tuned threshold for singularity detection
        
        # --- Discontinuity Detection (Auto-Reset on Reconnection) ---
        self.last_quat_parent = None  # Track last quaternion to detect sensor reconnections
        self.last_quat_child = None
        self.discontinuity_threshold = np.deg2rad(60.0)  # 60° jump = new connection

    def add_packet_and_optimize(self, r_par_aligned, r_chi_aligned):
        q_par = r_par_aligned.as_quat()
        q_chi = r_chi_aligned.as_quat()
        
        # --- AUTO-RESET ON SENSOR RECONNECTION ---
        # Detect discontinuities (large quaternion jumps > 60°) indicating sensor reconnection
        if self.last_quat_parent is not None:
            # Calculate angular distance between consecutive quaternions
            q_diff_par = R.from_quat(self.last_quat_parent).inv() * R.from_quat(q_par)
            q_diff_chi = R.from_quat(self.last_quat_child).inv() * R.from_quat(q_chi)
            # as_rotvec() gives rotation vector; magnitude = rotation angle in radians
            angle_dist_par = np.linalg.norm(q_diff_par.as_rotvec())
            angle_dist_chi = np.linalg.norm(q_diff_chi.as_rotvec())
            
            if angle_dist_par > self.discontinuity_threshold or angle_dist_chi > self.discontinuity_threshold:
                print(f"🔄 [2D] SENSOR RECONNECTION DETECTED! (Jump: {np.degrees(max(angle_dist_par, angle_dist_chi)):.1f}°) Resetting optimizer state...")
                self._reset_filter_state()
        
        self.last_quat_parent = q_par
        self.last_quat_child = q_chi
        
        self.buffer_parent.append(q_par)
        self.buffer_child.append(q_chi)
        
        if len(self.buffer_parent) >= self.window_size:
            if not self.is_calculating:
                buf_par_copy = self.buffer_parent.copy()
                buf_chi_copy = self.buffer_child.copy()
                
                t = threading.Thread(target=self._run_optimization_threaded, args=(buf_par_copy, buf_chi_copy))
                t.daemon = True
                t.start()
            
            keep_elements = max(0, self.window_size - self.step_size)
            self.buffer_parent = self.buffer_parent[-keep_elements:] if keep_elements > 0 else []
            self.buffer_child = self.buffer_child[-keep_elements:] if keep_elements > 0 else []
            
        # Always return the currently found target directly (no visual faking/smoothing)
        self.current_heading_offset = self.target_heading_offset
        return self.current_heading_offset, self.latest_angles

    def _reset_filter_state(self):
        """Reset all filter states to prevent unbounded drift on sensor reconnection."""
        self.b_w_minus_1 = 0.0
        self.delta_w_minus_1 = 0.0
        self.delta_f_w_minus_1 = 0.0
        self.seed_B_w_minus_1 = np.pi
        self.target_heading_offset = 0.0
        self.current_heading_offset = 0.0
        self.rom_violation_counter = 0
        self.buffer_parent = []
        self.buffer_child = []
    
    def _eval_limrom_cost(self, delta_yaw, r_parent_inv, r_child_window):
        orig_limrom = self.limrom_mode
        self.limrom_mode = "limrom_referee" # Temporär für Referee-Entscheidung aktivieren
        orig_weight = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        res = self._residuals([delta_yaw], r_parent_inv, r_child_window)
        cost = np.sum(res**2)
        self.limrom_mode = orig_limrom
        self.delta_delta_weight = orig_weight
        return cost

    def _residuals(self, delta_yaw_array, r_parent_inv, r_child_window):
        delta_yaw = delta_yaw_array[0]
        
        # 1. Apply yaw offset
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_child_window
        
        # 2. Calculate absolute joint orientation (relative quaternion)
        r_rel = r_parent_inv * r_child_corrected
        
        # 3. 2DoF Constraint formula from the paper
        q_scipy = r_rel.as_quat()
        x = q_scipy[:, 0] # q1
        y = q_scipy[:, 1] # q2
        z = q_scipy[:, 2] # q3
        w = q_scipy[:, 3] # q0
        
        # --- ALTE WINKELBERECHNUNG (auskommentiert) ---
        beta_0_hat = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        
        # --- KORREKTE TORSIONS-BERECHNUNG (Gimbal-Lock frei) ---
        #beta_0_hat = 2.0 * np.arctan2(z, w)
        
        # --- ANATOMICAL CONSTRAINTS (LimRoM PENALTY) ---
        angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        
        x_upper_bound = np.deg2rad(35.0)
        x_lower_bound = np.deg2rad(-35.0)
        y_upper_bound = np.deg2rad(140.0)
        y_lower_bound = np.deg2rad(-170.0)
        
        gl_mask = np.abs(np.abs(angle_y) - (np.pi / 2.0)) < 0.2
        
        penalty_over_x = np.where(gl_mask, 0.0, np.maximum(0, angle_x - x_upper_bound))
        penalty_under_x = np.where(gl_mask, 0.0, np.maximum(0, x_lower_bound - angle_x))
        penalty_over_y = np.maximum(0, angle_y - y_upper_bound)
        penalty_under_y = np.maximum(0, y_lower_bound - angle_y)
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        # LimRom ist standardmäßig deaktiviert, außer der User wünscht es explizit für ein Paper oder der Dual Seed Referee prüft die Täler!
        if self.limrom_mode in ["limrom_paper", "limrom_referee"] or getattr(self, 'enable_limrom', False):
            res_list.extend([penalty_over_x, penalty_under_x, penalty_over_y, penalty_under_y])
        if self.delta_delta_weight > 0.0:
            regularization_residual = (self.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.delta_delta_weight)
            res_list.append([regularization_residual])
            
        residuals = np.concatenate(res_list)
        
        # --- DEBUG LOGGING: Speichere jeden von least_squares getesteten Punkt ---
        if self.debug_log_file and hasattr(self, '_current_debug_info'):
            cost = np.sum(residuals**2)
            info = self._current_debug_info
            with open(self.debug_log_file, "a") as f:
                f.write(f"{time.time()},{info['w_index']},{np.degrees(delta_yaw):.4f},{cost:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")
        
        return residuals

    def _run_optimization_threaded(self, buf_par, buf_chi):
        self.is_calculating = True
        try:
            # --- STATIONARY STATE DETECTION ("FLAT VALLEY") ---
            # If the sensors are completely still, the optimization problem is underconstrained.
            # We detect this to freeze the update and prevent noise amplification.
            quat_diffs_par = np.diff(buf_par, axis=0)
            quat_diffs_chi = np.diff(buf_chi, axis=0)
            movement_var_par = np.sum(quat_diffs_par**2)
            movement_var_chi = np.sum(quat_diffs_chi**2)
            
            # Movement threshold: is it a flat valley?
            is_flat_valley = (movement_var_par < self.flat_valley_threshold and movement_var_chi < self.flat_valley_threshold)
            print(f"🔍 [2D] Variance Check | Parent: {movement_var_par:.6E}, Child: {movement_var_chi:.6E} | Threshold: {self.flat_valley_threshold:.6E}")
            
            r_parent = R.from_quat(buf_par)
            r_child = R.from_quat(buf_chi)
            r_parent_inv = r_parent.inv()
            
            # --- WINDOW RATING r_w ---
            j1_z = r_parent.apply([0, 1, 0])[:, 2] 
            j2_z = r_child.apply([1, 0, 0])[:, 2]  
            
            proj_j1 = np.sqrt(np.clip(1.0 - j1_z**2, 0.0, 1.0))
            proj_j2 = np.sqrt(np.clip(1.0 - j2_z**2, 0.0, 1.0))
            r_w_k = proj_j1 * proj_j2
            r_w = np.sqrt(np.mean(r_w_k**2))
            
            opt_start = time.time()
            
            # Logik: Soll die Optimierung übersprungen werden?
            should_bypass_opt = is_flat_valley and self.enable_flat_valley_filter
            
            if should_bypass_opt:
                delta_w = self.delta_w_minus_1
                opt_duration = time.time() - opt_start
                optimization_success = True
                cost_fun_val = 0.0
                
                # Write a single debug line for flat valley indicating no search occurred
                if self.debug_log_file:
                    with open(self.debug_log_file, "a") as f:
                        f.write(f"{time.time()},{self.w_index + 1},{np.degrees(delta_w):.4f},{cost_fun_val:.6f},1,{movement_var_par:.6E},{movement_var_chi:.6E},1,{r_w:.4f},{np.degrees(delta_w):.4f}\n")
            else:
                # --- LEVENBERG-MARQUARDT (GAUSS-NEWTON) OPTIMIZATION ---
                initial_guess = [self.delta_f_w_minus_1]
                # Grid Search in Frame 0: IMMER machen, um das richtige anatomische Tal zu finden!
                # Dazu temporär LimRom auf 'classic' schalten, falls wir in 'kinematic_constraints' sind.
                if self.w_index == 0:
                    best_cost_init = float('inf')
                    orig_limrom = getattr(self, 'limrom_mode', 'off')
                    self.limrom_mode = 'classic' # Temporär für den Scan aktivieren
                    for test_deg in range(-180, 180, 15):
                        test_rad = np.deg2rad(test_deg)
                        cost_init = np.sum(self._residuals([test_rad], r_parent_inv, r_child)**2)
                        if cost_init < best_cost_init:
                            best_cost_init = cost_init
                            initial_guess = [test_rad]
                    self.limrom_mode = orig_limrom # Zurücksetzen
                    print(f"🌍 [2D] Initial Grid Search abgeschlossen. Starte bei {np.degrees(initial_guess[0]):.1f}°")
                
                self._current_debug_info = {
                    'w_index': self.w_index + 1,
                    'var_up': movement_var_par,
                    'var_low': movement_var_chi,
                    'is_flat': int(is_flat_valley),
                    'r_w': r_w
                }
                
                # --- COARSE SEARCH FOR PLOTTING ---
                if self.debug_log_file:
                    info = self._current_debug_info
                    del self._current_debug_info
                    
                    seed_a = self.delta_f_w_minus_1
                    seed_b = self.seed_B_w_minus_1
                    
                    for test_deg in range(-180, 180, 5):
                        test_rad = np.deg2rad(test_deg)
                        res_coarse = self._residuals([test_rad], r_parent_inv, r_child)
                        cost_coarse = np.sum(res_coarse**2)
                        with open(self.debug_log_file, "a") as f:
                            f.write(f"{time.time()},{info['w_index']},{test_deg:.4f},{cost_coarse:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")
                    
                    cost_a_seed = np.sum(self._residuals([seed_a], r_parent_inv, r_child)**2)
                    cost_b_seed = np.sum(self._residuals([seed_b], r_parent_inv, r_child)**2)
                    
                    # LimRom Evaluation
                    lr_cost_a = self._eval_limrom_cost(seed_a, r_parent_inv, r_child)
                    lr_cost_b = self._eval_limrom_cost(seed_b, r_parent_inv, r_child)
                    
                    with open(self.debug_log_file, "a") as f:
                        f.write(f"{time.time()},{info['w_index']},{np.degrees(seed_a):.4f},{cost_a_seed:.6f},3,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},{lr_cost_a:.6f}\n")
                        f.write(f"{time.time()},{info['w_index']},{np.degrees(seed_b):.4f},{cost_b_seed:.6f},4,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},{lr_cost_b:.6f}\n")
                    self._current_debug_info = info

                self.latest_jump_event = False
                self.latest_seed_lost = False
                
                if self.mode_kinematic_constraints:
                    # --- PURE CONSTRAINT REFEREE (DUAL SEED 2D TRACKING) ---
                    seed_A = self.delta_f_w_minus_1
                    res_A = least_squares(self._residuals, [seed_A], args=(r_parent_inv, r_child), method='lm')
                    sol_A_norm = (res_A.x[0] + np.pi) % (2 * np.pi) - np.pi
                    
                    # Tracked Seed B laden
                    seed_B = self.seed_B_w_minus_1
                    res_B = least_squares(self._residuals, [seed_B], args=(r_parent_inv, r_child), method='lm')
                    sol_B_norm = (res_B.x[0] + np.pi) % (2 * np.pi) - np.pi
                    
                    # Kollisions-Prüfung: Wenn Seed B ins selbe Tal wie A gerutscht ist
                    dist_A_B = np.abs((sol_B_norm - sol_A_norm + np.pi) % (2 * np.pi) - np.pi)
                    if dist_A_B < np.deg2rad(90.0):
                        print(f"⚠️ [2D] Täler verschmolzen (Distanz {np.degrees(dist_A_B):.1f}°)! Starte Grid-Scan für Ausweich-Tal...")
                        best_coarse_cost = float('inf')
                        # Wir suchen einen neuen Seed B weit weg von A
                        for test_deg in range(-180, 180, 15):
                            test_rad = np.deg2rad(test_deg)
                            dist_to_A = np.abs((test_rad - sol_A_norm + np.pi) % (2 * np.pi) - np.pi)
                            if dist_to_A > np.deg2rad(90.0):
                                cost_coarse = np.sum(self._residuals([test_rad], r_parent_inv, r_child)**2)
                                if cost_coarse < best_coarse_cost:
                                    best_coarse_cost = cost_coarse
                                    seed_B = test_rad
                                    
                        # Neu-Optimierung mit dem gefundenen, sicheren Seed B
                        res_B = least_squares(self._residuals, [seed_B], args=(r_parent_inv, r_child), method='lm')
                        sol_B_norm = (res_B.x[0] + np.pi) % (2 * np.pi) - np.pi
                        
                    # State für den B-Tracker im nächsten Frame updaten
                    self.seed_B_w_minus_1 = sol_B_norm
                    
                    # Referee Costs
                    orig_limrom = self.limrom_mode
                    orig_weight = self.delta_delta_weight
                    self.limrom_mode = "off"
                    self.delta_delta_weight = 0.0
                    cost_L_A = self._eval_limrom_cost(sol_A_norm, r_parent_inv, r_child)
                    cost_L_B = self._eval_limrom_cost(sol_B_norm, r_parent_inv, r_child)
                    self.limrom_mode = orig_limrom
                    self.delta_delta_weight = orig_weight
                    
                    is_A_legal = (cost_L_A < 5.0) and (np.sum(res_A.fun**2) < 5.0)
                    is_B_legal = (cost_L_B < 5.0) and (np.sum(res_B.fun**2) < 5.0)
                    
                    # Wenn beide legal sind, priorisiere immer Seed A (verhindert unnötiges Springen)
                    if not hasattr(self, 'rom_violation_counter'): self.rom_violation_counter = 0
                    
                    if is_A_legal:
                        self.rom_violation_counter = max(0, self.rom_violation_counter - 1)
                        best_yaw = sol_A_norm
                        best_cost = np.sum(res_A.fun**2)
                    elif is_B_legal and not is_A_legal:
                        self.rom_violation_counter += 1
                        if self.rom_violation_counter >= 3:
                            best_yaw = sol_B_norm
                            best_cost = np.sum(res_B.fun**2)
                            self.latest_jump_event = True
                            self.rom_violation_counter = 0 
                        else:
                            best_yaw = sol_A_norm
                            best_cost = np.sum(res_A.fun**2)
                    else:
                        # Beide illegal: Bleibe bei A
                        self.rom_violation_counter = max(0, self.rom_violation_counter - 1)
                        best_yaw = sol_A_norm
                        best_cost = np.sum(res_A.fun**2)
                            
                    optimization_success = res_A.success or res_B.success
                else:
                    # --- LEVENBERG-MARQUARDT (STANDARD) ---
                    res = least_squares(
                        self._residuals, 
                        initial_guess, 
                        args=(r_parent_inv, r_child), 
                        method='lm'
                    )
                    best_yaw = res.x[0]
                    best_cost = res.cost * 2.0
                    optimization_success = res.success
                
                if self.debug_log_file:
                    with open(self.debug_log_file, "a") as f:
                        # Log the starting seed
                        f.write(f"{time.time()},{self.w_index + 1},{np.degrees(initial_guess[0]):.4f},0.0,3,{movement_var_par:.6E},{movement_var_chi:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(initial_guess[0]):.4f}\n")
                        # Log the actual optimized minimum
                        f.write(f"{time.time()},{self.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_par:.6E},{movement_var_chi:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
                opt_duration = time.time() - opt_start
                delta_w = (best_yaw + np.pi) % (2 * np.pi) - np.pi
                cost_fun_val = best_cost
            
            if optimization_success:
                self.w_index += 1
                w_idx = self.w_index
                
                # --- HEADING FILTER / BIAS EXTRAPOLATION ---
                if self.enable_singularity_filter:
                    is_singular_due_to_flat_valley = is_flat_valley and self.enable_flat_valley_filter
                    is_singular = (r_w < self.r_min) or is_singular_due_to_flat_valley
                    s_w = 0.0 if is_singular else r_w
                    if is_singular_due_to_flat_valley:
                        print(f"💤 [2D] Flat Valley ACTIVE! Arm held still. Using bias extrapolation.")
                    elif is_singular:
                        print(f"⚠️ [2D] Singularity filter ACTIVE! (Rating (r_w): {r_w:.3f} < {self.r_min}). Using bias extrapolation.")
                    
                    # --- STATE OVERRIDE (ANTI WIND-UP) ---
                    if is_singular and self.enable_anti_windup:
                        delta_w = (self.delta_f_w_minus_1 + self.b_w_minus_1 + np.pi) % (2 * np.pi) - np.pi
                else:
                    is_singular = False
                    s_w = 1.0
                
                # Calculate adaptive factors
                k_b_w =     max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                # Calculate new drift rate and filtered offset (Angle-Safe Version!)
                if w_idx == 1:
                    b_w = 0.0
                    delta_f_w = delta_w
                else:
                    diff_w_w1 = (delta_w - self.delta_w_minus_1 + np.pi) % (2 * np.pi) - np.pi
                    b_w = self.b_w_minus_1 + s_w * k_b_w * (diff_w_w1 - self.b_w_minus_1)
                    
                    diff_w_fw1 = (delta_w - self.delta_f_w_minus_1 + np.pi) % (2 * np.pi) - np.pi
                    delta_f_w = self.delta_f_w_minus_1 + b_w + s_w * k_delta_w * (diff_w_fw1 - b_w)
                
                delta_f_w = (delta_f_w + np.pi) % (2 * np.pi) - np.pi
                
                self.b_w_minus_1 = b_w
                self.delta_w_minus_1 = delta_w
                self.delta_f_w_minus_1 = delta_f_w
                
                self.target_heading_offset = delta_f_w
                self.current_heading_offset = self.target_heading_offset
                
                # Berechne die finalen Euler-Winkel (Schulter) der gefundenen "besten" Lösung
                rot_offset_best = R.from_euler('z', delta_f_w, degrees=False)
                r_rel_best = r_parent_inv * (rot_offset_best * r_child)
                avg_angles = np.mean(r_rel_best.as_euler('XYZ', degrees=True), axis=0)

                #self.latest_angles['x'] = avg_angles[0]
                #self.latest_angles['y'] = avg_angles[1]
                #self.latest_angles['z'] = avg_angles[2]
                quats = r_rel_best.as_quat()

                x = quats[:, 0]
                y = quats[:, 1]
                z = quats[:, 2]
                w = quats[:, 3]

                # 2. Manuelle Berechnung der Euler-Winkel (in Radiant)
                # X-Achse (oft als Roll bezeichnet)
                angle_x_rad = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

                # Y-Achse (oft als Pitch bezeichnet)
                # np.clip schützt davor, dass float-Ungenauigkeiten (z.B. 1.0000001) den Arkussinus abstürzen lassen
                angle_y_rad = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))

                # Z-Achse (oft als Yaw bezeichnet)
                # KORREKTUR: Torsion Gimbal-Lock-frei berechnen
                angle_z_rad = 2.0 * np.arctan2(z, w)

                # 3. Umwandlung von Radiant in Grad
                angle_x_deg = np.rad2deg(angle_x_rad)
                angle_y_deg = np.rad2deg(angle_y_rad)
                angle_z_deg = np.rad2deg(angle_z_rad)

                # 4. Mittelwert über das gesamte Fenster bilden (wie in deinem Original-Code)
                self.latest_angles['x'] = np.mean(angle_x_deg)
                self.latest_angles['y'] = np.mean(angle_y_deg)
                self.latest_angles['z'] = np.mean(angle_z_deg)

                print(f"\033[95m[{time.strftime('%H:%M:%S')}]\033[0m 2D Optimizer (Shoulder):")
                print(f"    -> Filter Pipeline: Raw LM: \033[96m{np.degrees(delta_w):.2f}°\033[0m | Filtered Out: \033[38;5;208m{np.degrees(delta_f_w):.2f}°\033[0m | Bias: \033[93m{np.degrees(b_w):.4f}°\033[0m")
                print(f"    -> Modus: \033[93m{getattr(self, 'limrom_mode', 'off')}\033[0m | Kinematic Cost (LM Error): \033[91m{cost_fun_val:.6f}\033[0m")
                print(f"    -> Gelenkwinkel / Angles (X/Y/Z): \033[92m{self.latest_angles['x']:.1f}°, {self.latest_angles['y']:.1f}°, {self.latest_angles['z']:.1f}°\033[0m")    
                print(f"    -> Visibility (\033[94mr_w: {r_w:.2f}\033[0m) | Singularity: \033[91m{is_singular}\033[0m | duration: {opt_duration:.3f}s")

                # Logging
                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"{time.time()},{w_idx},{r_w:.4f},{int(is_singular)},{np.degrees(delta_w):.4f},{np.degrees(b_w):.4f},{np.degrees(delta_f_w):.4f},{cost_fun_val:.6f},{opt_duration:.4f},{self.latest_angles['x']:.2f},{self.latest_angles['y']:.2f},{self.latest_angles['z']:.2f},{k_b_w:.6f},{k_delta_w:.6f},{int(self.latest_jump_event)},{int(self.latest_seed_lost)}\n")
            else:
                print("⚠️ 2D optimization failed for this window.")
        except Exception as e:
            print(f"Error in 2D Optimizer-Thread: {e}")
        finally:
            self.is_calculating = False