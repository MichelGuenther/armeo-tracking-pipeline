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
    def __init__(self, sensor_upper, sensor_lower, window_size=200, step_size=100, flat_valley_threshold=1e-4, enable_singularity_filter=True, enable_flat_valley_filter=True, enable_anti_windup=True, enable_valley_retry_validation=True, tau_b_=2.8, tau_delta_=0.7, delta_delta_weight=0.0, limrom_mode="kinematic_constraints", log_file="drift_log_1D.csv", debug_log_file=None):
        """
        Initializes the 1D optimizer for a hinge joint (e.g., elbow).
        
        Args:
            limrom_mode (str): Optimization mode:
                - "kinematic_constraints": KC only, grid search on frame 0, then LM
                - "classic": KC + LimRoM penalty, grid search on frame 0, then LM  
                - "dual_seed": 1x LM (KC+LimRoM), then evaluate mirror point
        """
        self.s_upper = sensor_upper
        self.s_lower = sensor_lower
        self.window_size = window_size
        self.step_size = step_size
        self.flat_valley_threshold = flat_valley_threshold
        self.enable_singularity_filter = enable_singularity_filter
        self.enable_flat_valley_filter = enable_flat_valley_filter
        self.enable_anti_windup = enable_anti_windup
        self.limrom_mode = limrom_mode
            
        self.rom_violation_counter = 0  # Hysteresis Counter
        self.delta_delta_weight = delta_delta_weight
        self.log_file = log_file
        self.debug_log_file = debug_log_file
        
        self.latest_angle_x = 0.0
        self.latest_jump_event = False
        self.latest_seed_lost = False

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
    
    def _eval_kinematic_cost(self, delta_yaw, r_upper_inv, r_lower_window):
        """Evaluate ONLY the Kinematic cost (angle_y, angle_z) at a given delta_yaw."""
        delta_yaw_val = delta_yaw if isinstance(delta_yaw, (int, float)) else delta_yaw[0]
        rot_offset = R.from_euler('z', delta_yaw_val, degrees=False)
        r_lower_corrected = rot_offset * r_lower_window
        r_rel = r_upper_inv * r_lower_corrected
        
        q = r_rel.as_quat()
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        angle_z = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        
        kc_cost = np.sum(angle_y**2) + np.sum(angle_z**2)
        return kc_cost
    
    def _eval_limrom_cost(self, delta_yaw, r_upper_inv, r_lower_window):
        """Evaluate ONLY the LimRoM penalty cost at a given delta_yaw."""
        delta_yaw_val = delta_yaw if isinstance(delta_yaw, (int, float)) else delta_yaw[0]
        rot_offset = R.from_euler('z', delta_yaw_val, degrees=False)
        r_lower_corrected = rot_offset * r_lower_window
        r_rel = r_upper_inv * r_lower_corrected
        
        q = r_rel.as_quat()
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        
        upper_bound = np.deg2rad(96.0)
        lower_bound = np.deg2rad(-60.0)
        
        penalty_over = np.maximum(0, angle_x - upper_bound) #* 2.0
        penalty_under = np.maximum(0, lower_bound - angle_x) #* 2.0
        
        limrom_cost = np.sum(penalty_over**2) + np.sum(penalty_under**2)
        return limrom_cost
    
    def _optimize_1d_kinematic_constraints(self, r_upper_inv, r_lower, r_w):
        """Optimize using Kinematic Constraints only (no LimRoM penalties)."""
        initial_guess = [self.delta_f_w_minus_1]
        
        # Grid search only on first frame
        if self.w_index == 0:
            best_cost_init = float('inf')
            for test_deg in range(-180, 180, 15):
                test_rad = np.deg2rad(test_deg)
                cost_init = np.sum(self._residuals([test_rad], r_upper_inv, r_lower, include_limrom_penalty=False)**2)
                if cost_init < best_cost_init:
                    best_cost_init = cost_init
                    initial_guess = [test_rad]
            print(f"🌍 [1D KC] Initial Grid Search (KC only). Starting at {np.degrees(initial_guess[0]):.1f}°")
        
        # Levenberg-Marquardt optimization with KC only
        res = least_squares(
            self._residuals,
            initial_guess,
            args=(r_upper_inv, r_lower, False),  # False = no LimRoM penalty
            method='lm'
        )
        
        delta_w = res.x[0]
        cost_val = res.cost * 2.0
        optimization_success = res.success
        
        return delta_w, cost_val, optimization_success
    
    def _optimize_1d_classic(self, r_upper_inv, r_lower, r_w):
        """Optimize using Kinematic Constraints + LimRoM Penalty."""
        initial_guess = [self.delta_f_w_minus_1]
        
        # Grid search only on first frame
        if self.w_index == 0:
            best_cost_init = float('inf')
            for test_deg in range(-180, 180, 15):
                test_rad = np.deg2rad(test_deg)
                cost_init = np.sum(self._residuals([test_rad], r_upper_inv, r_lower, include_limrom_penalty=True)**2)
                if cost_init < best_cost_init:
                    best_cost_init = cost_init
                    initial_guess = [test_rad]
            print(f"🌍 [1D Classic] Initial Grid Search (KC + LimRoM). Starting at {np.degrees(initial_guess[0]):.1f}°")
        
        # Levenberg-Marquardt optimization with KC + LimRoM
        res = least_squares(
            self._residuals,
            initial_guess,
            args=(r_upper_inv, r_lower, True),  # True = with LimRoM penalty
            method='lm'
        )
        
        delta_w = res.x[0]
        cost_val = res.cost * 2.0
        optimization_success = res.success
        
        return delta_w, cost_val, optimization_success
    
    def _optimize_1d_dual_seed(self, r_upper_inv, r_lower, r_w):
        """
        Dual-Seed Optimizer:
        1. Optimize with KC+LimRoM to find sol_A
        2. Evaluate mirror point sol_B (180° away) WITHOUT optimizing
        3. Compare LimRoM costs: choose best
        """
        # 1. Optimize in current valley with KC + LimRoM
        res_A = least_squares(
            self._residuals,
            [self.delta_f_w_minus_1],
            args=(r_upper_inv, r_lower, True),  # KC + LimRoM
            method='lm'
        )
        sol_A_norm = (res_A.x[0] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Mirror point (180° away) - evaluate ONLY, do NOT optimize
        sol_B_raw = sol_A_norm + np.pi
        sol_B_norm = sol_B_raw - 2 * np.pi if sol_B_raw > np.pi else sol_B_raw
        
        # 3. Compute costs at both points
        kc_cost_A = self._eval_kinematic_cost(sol_A_norm, r_upper_inv, r_lower)
        limrom_cost_A = self._eval_limrom_cost(sol_A_norm, r_upper_inv, r_lower)
        cost_total_A = kc_cost_A + limrom_cost_A
        
        kc_cost_B = self._eval_kinematic_cost(sol_B_norm, r_upper_inv, r_lower)
        limrom_cost_B = self._eval_limrom_cost(sol_B_norm, r_upper_inv, r_lower)
        cost_total_B = kc_cost_B + limrom_cost_B
        
        # 4. Decide based on LimRoM cost
        if limrom_cost_B < limrom_cost_A:
            self.rom_violation_counter += 1
            if self.rom_violation_counter >= 3:
                delta_w = sol_B_norm
                cost_val = cost_total_B
                self.latest_jump_event = True
                print(f"🔀 [1D Dual Seed] FLIP: Mirror point ({limrom_cost_B:.2f}) better than current ({limrom_cost_A:.2f})")
                self.rom_violation_counter = 0
            else:
                delta_w = sol_A_norm
                cost_val = cost_total_A
        else:
            self.rom_violation_counter = max(0, self.rom_violation_counter - 1)
            delta_w = sol_A_norm
            cost_val = cost_total_A
        
        # Debug logging: seed A (is_best=3) and seed B (is_best=4)
        # cost_val = KC-only cost so markers land on the KC landscape curve
        # best_yaw_deg = LimRoM cost shown as annotation in the detail plot
        if self.debug_log_file:
            info = self._current_debug_info if hasattr(self, '_current_debug_info') else {'var_up': 0.0, 'var_low': 0.0, 'is_flat': 0}
            with open(self.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.w_index + 1},{np.degrees(sol_A_norm):.4f},{kc_cost_A:.6f},3,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{r_w:.4f},{limrom_cost_A:.4f}\n")
                f.write(f"{time.time()},{self.w_index + 1},{np.degrees(sol_B_norm):.4f},{kc_cost_B:.6f},4,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{r_w:.4f},{limrom_cost_B:.4f}\n")
        
        optimization_success = True
        return delta_w, cost_val, optimization_success


    def _residuals(self, delta_yaw_array, r_upper_inv, r_lower_window, include_limrom_penalty=False):
        """
        Compute residuals for 1D hinge joint.
        
        Args:
            include_limrom_penalty: If True, add anatomical ROM penalties. Used for "classic" mode.
        """
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
        
        res_list = [angle_y, angle_z]
        
        # --- ANATOMICAL CONSTRAINTS (LimRoM PENALTY) ---
        # Only add penalties if include_limrom_penalty is True (used in "classic" mode)
        if include_limrom_penalty:
            penalty_over = np.maximum(0, angle_x - upper_bound) * 2.0
            penalty_under = np.maximum(0, lower_bound - angle_x) * 2.0
            res_list.append(penalty_over)
            res_list.append(penalty_under)
            
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
                # --- 1D OPTIMIZER MODE DISPATCHER ---
                self.latest_jump_event = False
                self.latest_seed_lost = False
                opt_start_inner = time.time()

                self._current_debug_info = {
                    'w_index': self.w_index + 1,
                    'var_up': movement_var_up,
                    'var_low': movement_var_low,
                    'is_flat': int(is_flat_valley),
                    'r_w': r_w
                }

                # --- COARSE SCAN FOR DEBUG PLOTTING ---
                # Sweeps -180..175° in 5° steps to visualize the full cost landscape.
                # _current_debug_info is temporarily removed so _residuals doesn't double-log.
                if self.debug_log_file:
                    info = self._current_debug_info
                    del self._current_debug_info
                    include_limrom = self.limrom_mode == "classic"
                    for test_deg in range(-180, 180, 5):
                        test_rad = np.deg2rad(test_deg)
                        res_coarse = self._residuals([test_rad], r_upper_inv, r_lower, include_limrom)
                        cost_coarse = np.sum(res_coarse**2)
                        with open(self.debug_log_file, "a") as f:
                            f.write(f"{time.time()},{info['w_index']},{test_deg:.4f},{cost_coarse:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")
                    # For dual_seed, don't restore: LM uses KC+LimRoM but landscape is KC-only
                    if self.limrom_mode != "dual_seed":
                        self._current_debug_info = info

                self.latest_jump_event = False
                self.latest_seed_lost = False

                if self.limrom_mode == "kinematic_constraints":
                    delta_w, cost_fun_val, optimization_success = self._optimize_1d_kinematic_constraints(r_upper_inv, r_lower, r_w)
                elif self.limrom_mode == "classic":
                    delta_w, cost_fun_val, optimization_success = self._optimize_1d_classic(r_upper_inv, r_lower, r_w)
                elif self.limrom_mode == "dual_seed":
                    delta_w, cost_fun_val, optimization_success = self._optimize_1d_dual_seed(r_upper_inv, r_lower, r_w)
                else:
                    # Fallback to kinematic constraints
                    print(f"⚠️ Unknown limrom_mode: {self.limrom_mode}. Falling back to kinematic_constraints.")
                    delta_w, cost_fun_val, optimization_success = self._optimize_1d_kinematic_constraints(r_upper_inv, r_lower, r_w)

                # Write the selected minimum for all modes (dual_seed writes seeds separately)
                if self.debug_log_file:
                    with open(self.debug_log_file, "a") as f:
                        f.write(f"{time.time()},{self.w_index + 1},{np.degrees(delta_w):.4f},{cost_fun_val:.6f},1,{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(delta_w):.4f}\n")

                if hasattr(self, '_current_debug_info'):
                    del self._current_debug_info

                opt_duration = time.time() - opt_start_inner
            
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
                
                # --- VALLEY JUMP: reset filter history to avoid bias corruption ---
                if self.latest_jump_event:
                    self.b_w_minus_1 = 0.0
                    self.delta_w_minus_1 = delta_w
                    self.delta_f_w_minus_1 = delta_w
                    print(f"🦘 [1D] Valley Jump! Filter history reset to {np.degrees(delta_w):.1f}°")

                # Filter smoothing parameters based on time constants
                k_b_w =     max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)

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
    def __init__(self, sensor_parent, sensor_child, window_size=200, step_size=100, flat_valley_threshold=1e-8, enable_singularity_filter=True, enable_flat_valley_filter=True, enable_anti_windup=True, enable_valley_retry_validation=True, tau_b_=2.8, tau_delta_=0.7, delta_delta_weight=0.0, adaptive_delta_weight_2d=0.0, limrom_mode="kinematic_constraints", log_file="drift_log_2D.csv", debug_log_file=None):
        """
        Initializes the 2D optimizer for a universal joint (e.g., shoulder).
        
        Args:
            limrom_mode (str): Optimization mode:
                - "kinematic_constraints": Torsion only (angle_z), grid search on frame 0, then LM. No ROM penalties.
                - "classic": Torsion + ROM penalties, grid search on frame 0, then LM
                - "dual_seed": 2x LM (two valleys), LimRoM referee decides
        """
        self.s_parent = sensor_parent
        self.s_child = sensor_child
        self.window_size = window_size
        self.step_size = step_size
        self.flat_valley_threshold = flat_valley_threshold
        self.enable_singularity_filter = enable_singularity_filter
        self.enable_flat_valley_filter = enable_flat_valley_filter
        self.enable_anti_windup = enable_anti_windup
        self.delta_delta_weight = delta_delta_weight
        self.adaptive_delta_weight_2d = adaptive_delta_weight_2d
        self.log_file = log_file
        self.debug_log_file = debug_log_file

        self.limrom_mode = limrom_mode

        self.rom_violation_counter = 0
        self.jump_cooldown_counter = 0
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
        self.jump_cooldown_counter = 0
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
        
        # --- TORSION (angle_z / forbidden axis) ---
        beta_0_hat = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        # --- ANATOMICAL CONSTRAINTS (LimRoM PENALTY) ---
        # Add penalties only for "classic" and "dual_seed" modes
        if self.limrom_mode in ["classic", "dual_seed"]:
            angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
            
            x_upper_bound = np.deg2rad(35.0)
            x_lower_bound = np.deg2rad(-20.0)
            y_upper_bound = np.deg2rad(140.0)
            y_lower_bound = np.deg2rad(-180.0)
            
            gl_mask = np.abs(np.abs(angle_y) - (np.pi / 2.0)) < 0.2
            
            penalty_over_x = np.where(gl_mask, 0.0, np.maximum(0, angle_x - x_upper_bound))
            penalty_under_x = np.where(gl_mask, 0.0, np.maximum(0, x_lower_bound - angle_x))
            penalty_over_y = np.maximum(0, angle_y - y_upper_bound)
            penalty_under_y = np.maximum(0, y_lower_bound - angle_y)
            
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
    
    def _eval_torsion_cost(self, delta_yaw, r_parent_inv, r_child_window):
        """Evaluate ONLY the Torsion cost (angle_z) at a given delta_yaw."""
        delta_yaw_val = delta_yaw if isinstance(delta_yaw, (int, float)) else delta_yaw[0]
        rot_offset = R.from_euler('z', delta_yaw_val, degrees=False)
        r_child_corrected = rot_offset * r_child_window
        r_rel = r_parent_inv * r_child_corrected
        
        q = r_rel.as_quat()
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        beta_0_hat = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        torsion_cost = np.sum(beta_0_hat**2)
        return torsion_cost
    
    def _eval_limrom_cost_2d(self, delta_yaw, r_parent_inv, r_child_window):
        """Evaluate ONLY the LimRoM penalty cost at a given delta_yaw."""
        delta_yaw_val = delta_yaw if isinstance(delta_yaw, (int, float)) else delta_yaw[0]
        rot_offset = R.from_euler('z', delta_yaw_val, degrees=False)
        r_child_corrected = rot_offset * r_child_window
        r_rel = r_parent_inv * r_child_corrected
        
        q = r_rel.as_quat()
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        
        x_upper_bound = np.deg2rad(35.0)
        x_lower_bound = np.deg2rad(-20.0)
        y_upper_bound = np.deg2rad(140.0)
        y_lower_bound = np.deg2rad(-180.0)
        
        gl_mask = np.abs(np.abs(angle_y) - (np.pi / 2.0)) < 0.2
        
        penalty_over_x = np.where(gl_mask, 0.0, np.maximum(0, angle_x - x_upper_bound)) * 2.0
        penalty_under_x = np.where(gl_mask, 0.0, np.maximum(0, x_lower_bound - angle_x)) * 2.0
        penalty_over_y = np.maximum(0, angle_y - y_upper_bound) * 2.0
        penalty_under_y = np.maximum(0, y_lower_bound - angle_y) * 2.0
        
        limrom_cost = np.sum(penalty_over_x**2) + np.sum(penalty_under_x**2) + np.sum(penalty_over_y**2) + np.sum(penalty_under_y**2)
        return limrom_cost
    
    def _optimize_2d_kinematic_constraints(self, r_parent_inv, r_child, r_w):
        """Optimize using Torsion only (no LimRoM penalties)."""
        initial_guess = [self.delta_f_w_minus_1]
        
        # Grid search only on first frame
        if self.w_index == 0:
            best_cost_init = float('inf')
            for test_deg in range(-180, 180, 15):
                test_rad = np.deg2rad(test_deg)
                cost_init = np.sum(self._residuals([test_rad], r_parent_inv, r_child)**2)
                if cost_init < best_cost_init:
                    best_cost_init = cost_init
                    initial_guess = [test_rad]
            print(f"🌍 [2D KC] Initial Grid Search (Torsion only). Starting at {np.degrees(initial_guess[0]):.1f}°")
        
        # Levenberg-Marquardt optimization with torsion only
        res = least_squares(
            self._residuals,
            initial_guess,
            args=(r_parent_inv, r_child),
            method='lm'
        )
        
        delta_w = res.x[0]
        cost_val = res.cost * 2.0
        optimization_success = res.success
        
        return delta_w, cost_val, optimization_success
    
    def _optimize_2d_classic(self, r_parent_inv, r_child, r_w):
        """Optimize using Torsion + LimRoM Penalty."""
        initial_guess = [self.delta_f_w_minus_1]
        
        # Save current mode and switch to classic (enables LimRoM penalties)
        orig_mode = self.limrom_mode
        self.limrom_mode = "classic"
        
        # Grid search only on first frame
        if self.w_index == 0:
            best_cost_init = float('inf')
            for test_deg in range(-180, 180, 15):
                test_rad = np.deg2rad(test_deg)
                cost_init = np.sum(self._residuals([test_rad], r_parent_inv, r_child)**2)
                if cost_init < best_cost_init:
                    best_cost_init = cost_init
                    initial_guess = [test_rad]
            print(f"🌍 [2D Classic] Initial Grid Search (Torsion + LimRoM). Starting at {np.degrees(initial_guess[0]):.1f}°")
        
        # Levenberg-Marquardt optimization with torsion + LimRoM
        res = least_squares(
            self._residuals,
            initial_guess,
            args=(r_parent_inv, r_child),
            method='lm'
        )
        
        delta_w = res.x[0]
        cost_val = res.cost * 2.0
        optimization_success = res.success
        
        self.limrom_mode = orig_mode
        
        return delta_w, cost_val, optimization_success
    
    def _optimize_2d_dual_seed(self, r_parent_inv, r_child, r_w, debug_info=None):
        """
        Dual-Seed Optimizer for 2D:
        Both seeds are optimized with KC only (torsion, no LimRoM in residuals).
        LimRoM is evaluated separately after LM and added to the KC cost.
        Referee: winner = lower total cost (KC + LimRoM).
        Collision guard: if the losing seed drifts within 45° of the winner,
        it is relocated via grid search to at least 90° away and re-optimized.
        """
        # Both LM runs use KC (torsion) only — LimRoM is evaluated separately afterwards.
        orig_mode = self.limrom_mode
        self.limrom_mode = "kinematic_constraints"

        seed_A = self.delta_f_w_minus_1
        res_A = least_squares(self._residuals, [seed_A], args=(r_parent_inv, r_child), method='lm')
        sol_A_norm = (res_A.x[0] + np.pi) % (2 * np.pi) - np.pi
        kc_cost_A = np.sum(res_A.fun**2)

        # Seed B must not be regularized — starts ~180° from delta_f_w_minus_1, so penalty would be ~π²×weight
        _weight_before_b = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        seed_B = self.seed_B_w_minus_1
        res_B = least_squares(self._residuals, [seed_B], args=(r_parent_inv, r_child), method='lm')
        sol_B_norm = (res_B.x[0] + np.pi) % (2 * np.pi) - np.pi
        kc_cost_B = np.sum(res_B.fun**2)
        self.delta_delta_weight = _weight_before_b

        self.limrom_mode = orig_mode

        # Collision guard: if seeds are within 45° of each other, relocate the inactive one.
        dist_A_B = np.abs((sol_B_norm - sol_A_norm + np.pi) % (2 * np.pi) - np.pi)
        if dist_A_B < np.deg2rad(45.0):
            print(f"⚠️ [2D] Seeds verschmolzen (Distanz {np.degrees(dist_A_B):.1f}°)! Grid-Scan für Ausweich-Tal...")
            best_coarse_cost = float('inf')
            for test_deg in range(-180, 180, 15):
                test_rad = np.deg2rad(test_deg)
                dist_to_A = np.abs((test_rad - sol_A_norm + np.pi) % (2 * np.pi) - np.pi)
                if dist_to_A > np.deg2rad(90.0):
                    cost_coarse = self._eval_torsion_cost(test_rad, r_parent_inv, r_child)
                    if cost_coarse < best_coarse_cost:
                        best_coarse_cost = cost_coarse
                        seed_B = test_rad

            self.limrom_mode = "kinematic_constraints"
            self.delta_delta_weight = 0.0
            res_B = least_squares(self._residuals, [seed_B], args=(r_parent_inv, r_child), method='lm')
            self.delta_delta_weight = _weight_before_b
            sol_B_norm = (res_B.x[0] + np.pi) % (2 * np.pi) - np.pi
            kc_cost_B = np.sum(res_B.fun**2)
            self.limrom_mode = orig_mode

        # Evaluate LimRoM cost separately (not part of optimization)
        lr_cost_A = self._eval_limrom_cost_2d(sol_A_norm, r_parent_inv, r_child)
        lr_cost_B = self._eval_limrom_cost_2d(sol_B_norm, r_parent_inv, r_child)

        total_cost_A = kc_cost_A + lr_cost_A
        total_cost_B = kc_cost_B + lr_cost_B

        # Debug: log both seeds on the torsion landscape; LimRoM cost as annotation
        if self.debug_log_file and debug_info is not None:
            with open(self.debug_log_file, "a") as f:
                f.write(f"{time.time()},{debug_info['w_index']},{np.degrees(sol_A_norm):.4f},{kc_cost_A:.6f},3,{debug_info['var_up']:.6E},{debug_info['var_low']:.6E},{debug_info['is_flat']},{debug_info['r_w']:.4f},{lr_cost_A:.4f}\n")
                f.write(f"{time.time()},{debug_info['w_index']},{np.degrees(sol_B_norm):.4f},{kc_cost_B:.6f},4,{debug_info['var_up']:.6E},{debug_info['var_low']:.6E},{debug_info['is_flat']},{debug_info['r_w']:.4f},{lr_cost_B:.4f}\n")

        # --- REFEREE (two-stage) ---

        # Cooldown: after a jump, freeze switching for N frames to prevent immediate back-jump.
        if self.jump_cooldown_counter > 0:
            self.jump_cooldown_counter -= 1
            delta_w = sol_A_norm
            cost_fun_val = kc_cost_A
            self.seed_B_w_minus_1 = sol_B_norm
            optimization_success = res_A.success or res_B.success
            return delta_w, cost_fun_val, optimization_success

        # Stage 1 – KC gate: B must have a comparable kinematic fit (max 2× worse than A).
        # If A has a much tighter torsion minimum, LimRoM is irrelevant — stay with A.
        kc_gate_passed = kc_cost_B <= kc_cost_A * 2.0

        if not kc_gate_passed:
            self.rom_violation_counter = max(0, self.rom_violation_counter - 1)
            delta_w = sol_A_norm
            cost_fun_val = kc_cost_A
            self.seed_B_w_minus_1 = sol_B_norm
        else:
            # Stage 2 – LimRoM tiebreaker: only reached when KC costs are comparable.
            # B must be at least 25% cheaper in total cost to count as a win.
            b_wins_this_frame = total_cost_B < total_cost_A * 0.75

            if not b_wins_this_frame:
                self.rom_violation_counter = max(0, self.rom_violation_counter - 1)
                delta_w = sol_A_norm
                cost_fun_val = kc_cost_A
                self.seed_B_w_minus_1 = sol_B_norm
            else:
                self.rom_violation_counter += 1
                if self.rom_violation_counter >= 3:
                    delta_w = sol_B_norm
                    cost_fun_val = kc_cost_B
                    self.latest_jump_event = True
                    self.jump_cooldown_counter = 10
                    self.seed_B_w_minus_1 = sol_A_norm  # old A becomes new B after role-swap
                    self.rom_violation_counter = 0
                    print(f"🦘 [2D] Valley Jump A→B | KC: {kc_cost_A:.4f}→{kc_cost_B:.4f} | LimRoM: {lr_cost_A:.1f}→{lr_cost_B:.1f}")
                else:
                    delta_w = sol_A_norm
                    cost_fun_val = kc_cost_A
                    self.seed_B_w_minus_1 = sol_B_norm

        optimization_success = res_A.success or res_B.success
        return delta_w, cost_fun_val, optimization_success

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
                self._current_debug_info = {
                    'w_index': self.w_index + 1,
                    'var_up': movement_var_par,
                    'var_low': movement_var_chi,
                    'is_flat': int(is_flat_valley),
                    'r_w': r_w
                }
                
                # --- COARSE SEARCH FOR PLOTTING ---
                # Always use pure torsion cost so both valleys are visible in the plot,
                # regardless of whether LimRoM is active in the actual optimizer.
                _dual_seed_debug_info = None
                if self.debug_log_file:
                    info = self._current_debug_info
                    del self._current_debug_info

                    for test_deg in range(-180, 180, 5):
                        test_rad = np.deg2rad(test_deg)
                        cost_coarse = self._eval_torsion_cost(test_rad, r_parent_inv, r_child)
                        with open(self.debug_log_file, "a") as f:
                            f.write(f"{time.time()},{info['w_index']},{test_deg:.4f},{cost_coarse:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")

                    if self.limrom_mode == "dual_seed":
                        # Pass info to the mode method so it can log sol_A/sol_B after LM.
                        # Do NOT restore _current_debug_info: the dual LM calls use torsion+LimRoM
                        # which would pollute the torsion-only landscape.
                        _dual_seed_debug_info = info
                    else:
                        # For KC/classic: restore so LM iterations are logged on the landscape.
                        # KC uses torsion only → consistent. Classic uses torsion+LimRoM but there
                        # is only one valley so mixing is acceptable.
                        self._current_debug_info = info

                self.latest_jump_event = False
                self.latest_seed_lost = False

                _orig_delta_weight = self.delta_delta_weight
                if self.adaptive_delta_weight_2d > 0 and r_w > 0:
                    self.delta_delta_weight = min(self.adaptive_delta_weight_2d / (r_w ** 2), 20.0)

                # --- 2D OPTIMIZER MODE DISPATCHER ---
                if self.limrom_mode == "kinematic_constraints":
                    delta_w, cost_fun_val, optimization_success = self._optimize_2d_kinematic_constraints(r_parent_inv, r_child, r_w)

                elif self.limrom_mode == "classic":
                    delta_w, cost_fun_val, optimization_success = self._optimize_2d_classic(r_parent_inv, r_child, r_w)

                elif self.limrom_mode == "dual_seed":
                    delta_w, cost_fun_val, optimization_success = self._optimize_2d_dual_seed(r_parent_inv, r_child, r_w, _dual_seed_debug_info)

                else:
                    # Fallback zu kinematic_constraints
                    print(f"⚠️ Unknown limrom_mode: {self.limrom_mode}. Falling back to kinematic_constraints.")
                    delta_w, cost_fun_val, optimization_success = self._optimize_2d_kinematic_constraints(r_parent_inv, r_child, r_w)

                self.delta_delta_weight = _orig_delta_weight

                if hasattr(self, '_current_debug_info'):
                    del self._current_debug_info

                if self.debug_log_file:
                    with open(self.debug_log_file, "a") as f:
                        f.write(f"{time.time()},{self.w_index + 1},{np.degrees(delta_w):.4f},{cost_fun_val:.6f},1,{movement_var_par:.6E},{movement_var_chi:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(delta_w):.4f}\n")
                
                opt_duration = time.time() - opt_start
                delta_w = (delta_w + np.pi) % (2 * np.pi) - np.pi
            
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
                
                # --- VALLEY JUMP: reset filter history to avoid bias corruption ---
                # A valley switch is not a drift event. Without reset, diff_w_w1 ≈ ±180°
                # which would give the bias filter a massive kick in the wrong direction.
                if self.latest_jump_event:
                    self.b_w_minus_1 = 0.0
                    self.delta_w_minus_1 = delta_w
                    self.delta_f_w_minus_1 = delta_w
                    print(f"🦘 [2D] Valley Jump! Filter history reset to {np.degrees(delta_w):.1f}°")

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

                # Intrinsic XZY decomposition: X and Y both get full (-180°, 180°] range.
                # Z (torsion, the constrained axis) is the middle angle, limited to [-90°, 90°],
                # which is acceptable since it should be near 0 after optimization.
                euler_xzy = r_rel_best.as_euler('XZY', degrees=True)  # shape (N, 3): [X, Z, Y]
                self.latest_angles['x'] = np.mean(euler_xzy[:, 0])   # flexion/extension
                self.latest_angles['z'] = np.mean(euler_xzy[:, 1])   # torsion
                self.latest_angles['y'] = np.mean(euler_xzy[:, 2])   # abduction/adduction

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