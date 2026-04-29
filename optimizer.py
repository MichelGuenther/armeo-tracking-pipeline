import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R
import time
import threading

class Optimizer1D:
    """
    This optimizer implements the 1D joint constraints for a hinge joint (e.g., elbow).
    Since the magnetometers are disabled, the IMUs accumulate a heading drift (yaw drift) around the global Z-axis.
    Assumption: The elbow (as a hinge joint) anatomically allows only one axis of rotation (flexion/extension). 
    Movements on the other two axes (abduction/adduction and internal/external rotation) are anatomical impossibilities (or "forbidden" axes).
    The optimizer collects data in a sliding window and searches for the heading correction angle (delta_yaw) 
    that minimizes the measured movement (variance) on these two "forbidden" axes.
    """
    def __init__(self, sensor_upper, sensor_lower, window_size=200, step_size=100, flat_valley_threshold=1e-4, enable_singularity_filter=True, log_file="drift_log_1D.csv"):
        """
        Initializes the 1D optimizer for a sensor pair.
        
        Args:
            sensor_upper (str): String ID of the upper arm sensor (parent).
            sensor_lower (str): String ID of the forearm sensor (child).
            window_size (int): Number of samples for the optimization window.
            step_size (int): After how many new samples the calculation should be *repeated*.
                             If step_size = window_size, there is no overlap (tiles back-to-back).
            flat_valley_threshold: Threshold (variance) from which movement is sufficient for finding a minimum.
        """
        self.s_upper = sensor_upper
        self.s_lower = sensor_lower
        self.window_size = window_size
        self.step_size = step_size
        self.flat_valley_threshold = flat_valley_threshold
        self.enable_singularity_filter = enable_singularity_filter
        self.log_file = log_file
        
        if self.log_file:
            import os
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write("time,window_index,r_w,is_singular,delta_w,b_w,delta_f_w,cost_val\n")
        
        # Buffers for the sliding window
        self.buffer_upper = []
        self.buffer_lower = []
        
        self.is_calculating = False # Prevents thread traffic jams
        
        # Target offset from the optimizer and the smoothed current offset
        self.target_heading_offset = 0.0
        self.current_heading_offset = 0.0
        
        # --- Heading Filter States (Paper Eq. 15-20) ---
        self.w_index = 0
        self.b_w_minus_1 = 0.0
        self.delta_w_minus_1 = 0.0
        self.delta_f_w_minus_1 = 0.0
        self.T_s = step_size / 100.0  # Window duration in sec (Assumption: 100 Hz DataFrame)
        self.tau_b = 2.8              # Tunable time constant for bias filter
        self.tau_delta = 0.7          # Tunable time constant for heading filter
        self.r_min = 0.1              # Empirically tuned threshold for singularity detection

    def add_packet_and_optimize(self, r_up_aligned, r_low_aligned):
        """
        Takes aligned sensor rotations (SciPy) from the manager/bridge,
        fills the sliding window and triggers the asynchronous optimization once the window is full.
        """
        self.buffer_upper.append(r_up_aligned.as_quat())
        self.buffer_lower.append(r_low_aligned.as_quat())
        
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
        return self.current_heading_offset
    '''
    def _cost_function(self, delta_yaw, r_upper_inv, r_lower_window):
        # 1. Heading-Offset auf das untere Gelenk anwenden
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_lower_corrected = rot_offset * r_lower_window
        
        # 2. Relative Orientierung zwischen Ober- und Unterarm berechnen
        r_rel = r_upper_inv * r_lower_corrected
        
        # 3. Quaternionen extrahieren
        q = r_rel.as_quat()
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # --- EULER WINKEL EXTRAKTION (vgl. Eq. 13 im Paper) ---
        angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        angle_z = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        
        # --- DEINE SPEZIFISCHEN GELENKGRENZEN ---
        upper_bound = np.deg2rad(98.0)
        lower_bound = np.deg2rad(-28.0)
        
        # --- ROBUSTER LimRoM-PENALTY (Quadratische Parabel für SciPy) ---
        # Diese quadratische Strafe sorgt dafür, dass der Optimizer den Gradienten 
        # (die Steigung) spürt und nicht blind in der 180-Grad-Mehrdeutigkeit stecken bleibt.
        penalty_over = np.sum(np.maximum(0, angle_x - upper_bound)**2)
        penalty_under = np.sum(np.maximum(0, lower_bound - angle_x)**2)
        
        # Gewichtung der Strafe (1000.0 zwingt SciPy sehr strikt in den erlaubten Bereich)
        limrom_penalty = (penalty_under + penalty_over) * 1000.0
        
        # --- GESAMTKOSTEN ---
        # Wir minimieren die Bewegungen auf den "verbotenen" Achsen (Y und Z) 
        # und addieren die Strafe, falls die X-Achse (Beugung) anatomisch unmöglich wird.
        cost = np.sum(angle_y**2 + angle_z**2) + limrom_penalty
        
        return cost
    '''
    def _cost_function(self, delta_yaw, r_upper_inv, r_lower_window):
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
        lower_bound = np.deg2rad(-55.0)
        
        # --- ANATOMICAL CONSTRAINTS (LimRoM PENALTY) ---
        # Excludes mirrored or anatomically impossible orientations (the "second valley")
        # by strictly penalizing states outside valid joint bounds (cf. Eq. 14).
        is_invalid = (angle_x > upper_bound) | (angle_x < lower_bound)
        e_k = is_invalid.astype(float)
        
        # --- COST FUNCTION (cf. Eq. 17) ---
        limrom_penalty = np.sum(e_k)
        
        cost = np.sum(angle_y**2 + angle_z**2) + limrom_penalty
        
        return cost

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
            
            # --- STATIONARY PERIOD BYPASS ("FLAT VALLEY") ---
            if is_flat_valley:
                # We skip the search entirely and freeze the old value update 
                # (bias will still extrapolate) to protect the model from noise.
                delta_w = self.delta_w_minus_1
                opt_duration = time.time() - opt_start
                optimization_success = True
                cost_fun_val = 0.0  # For the logger
            else:
                # --- COARSE-TO-FINE GRID SEARCH ---
                # Search using coarse steps
                coarse_space = np.linspace(-np.pi, np.pi, 36)
                best_coarse_cost = float('inf')
                best_coarse_yaw = 0.0
                
                for yaw in coarse_space:
                    cost = self._cost_function(yaw, r_upper_inv, r_lower)
                    if cost < best_coarse_cost:
                        best_coarse_cost = cost
                        best_coarse_yaw = yaw
                        
                # Search fine grid around best coarse result
                fine_radius = np.deg2rad(10.0)
                fine_space = np.linspace(best_coarse_yaw - fine_radius, best_coarse_yaw + fine_radius, 20)
                
                best_cost = best_coarse_cost
                best_yaw = best_coarse_yaw
                
                for yaw in fine_space:
                    cost = self._cost_function(yaw, r_upper_inv, r_lower)
                    if cost < best_cost:
                        best_fine_cost = cost
                        best_fine_yaw = yaw
                        
                # Search ultra-fine grid around best fine result
                ultra_fine_radius = np.deg2rad(1.0)
                ultra_fine_space = np.linspace(best_fine_yaw - ultra_fine_radius, best_fine_yaw + ultra_fine_radius, 21)
                best_cost = best_fine_cost
                best_yaw = best_fine_yaw

                for yaw in ultra_fine_space:
                    cost = self._cost_function(yaw, r_upper_inv, r_lower)
                    if cost < best_cost:
                        best_cost = cost
                        best_yaw = yaw
                
                opt_duration = time.time() - opt_start
                optimization_success = True 
                delta_w = best_yaw
                cost_fun_val = best_cost
            
            if optimization_success:
                self.w_index += 1
                w_idx = self.w_index
                
                # --- FILTER GAINS ---
                k_b_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                # --- HEADING FILTER / BIAS EXTRAPOLATION ---
                if self.enable_singularity_filter:
                    is_singular = (r_w < self.r_min) or is_flat_valley
                    s_w = 0.0 if is_singular else r_w
                    
                    if is_flat_valley:
                        print(f"💤 [1D] Flat Valley ACTIVE! Arm held still. Using bias extrapolation.")
                    elif is_singular:
                        print(f"⚠️ [1D] Singularity filter ACTIVE! (Rating (r_w): {r_w:.3f} < {self.r_min}). Using bias extrapolation.")
                else:
                    is_singular = False
                    s_w = 1.0 # Trust fully without dampening
                
                # Update bias (learned yaw drift rate)
                b_w = self.b_w_minus_1 + s_w * k_b_w * (delta_w - self.delta_w_minus_1 - self.b_w_minus_1)
                
                # Filtered heading offset (drift correction)
                delta_f_w = self.delta_f_w_minus_1 + b_w + s_w * k_delta_w * (delta_w - self.delta_f_w_minus_1 - b_w)
                
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
                avg_angle_x_deg = np.degrees(np.mean(best_angle_x))
                
                print(f"[{time.strftime('%H:%M:%S')}] 1D Optimizer (Grid): Target offset = {np.degrees(self.current_heading_offset):.2f}° (r_w: {r_w:.2f}; duration: {opt_duration:.3f}s; angle_x (Beugung): {avg_angle_x_deg:.1f}°)")
                
                # Logging
                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"{time.time()},{w_idx},{r_w:.4f},{int(is_singular)},{np.degrees(delta_w):.4f},{np.degrees(b_w):.4f},{np.degrees(delta_f_w):.4f},{cost_fun_val:.6f}\n")
                        
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
    def __init__(self, sensor_parent, sensor_child, window_size=200, step_size=100, flat_valley_threshold=1e-8, enable_singularity_filter=True, log_file="drift_log_2D.csv"):
        self.s_parent = sensor_parent
        self.s_child = sensor_child
        self.window_size = window_size
        self.step_size = step_size
        self.flat_valley_threshold = flat_valley_threshold
        self.enable_singularity_filter = enable_singularity_filter
        self.log_file = log_file
        
        if self.log_file:
            import os
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write("time,window_index,r_w,is_singular,delta_w,b_w,delta_f_w,cost_val\n")
        
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
        self.T_s = step_size / 100.0  # Window duration in sec (Assumption: 100 Hz)
        self.tau_b = 2.0              # Time constant bias (Eq. 19)
        self.tau_delta = 0.5          # Time constant heading (Eq. 20)
        self.r_min = 0.1              # Empirically tuned threshold for singularity detection

    def add_packet_and_optimize(self, r_par_aligned, r_chi_aligned):
        self.buffer_parent.append(r_par_aligned.as_quat())
        self.buffer_child.append(r_chi_aligned.as_quat())
        
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
        return self.current_heading_offset

    def _cost_function(self, delta_yaw, r_parent_inv, r_child_window):
        # 1. Apply yaw offset
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_child_window
        
        # 2. Calculate absolute joint orientation (relative quaternion)
        # Corresponds to relative joint orientation (cf. Equation 2 in the paper)
        r_rel = r_parent_inv * r_child_corrected
        
        # 3. 2DoF Constraint formula from the paper (cf. Equations 4 & 6 for the forbidden axis):
        # SciPy provides q in format [x, y, z, w]. The paper uses [q0, q1, q2, q3] with q0 = w.
        q_scipy = r_rel.as_quat()
        x = q_scipy[:, 0] # q1
        y = q_scipy[:, 1] # q2
        z = q_scipy[:, 2] # q3
        w = q_scipy[:, 3] # q0
        
        # Axial rotation (twist) is mapped to the Z-axis in this model.
        # The corresponding angle derivation using quaternion relations is:
        # beta_0_hat = arcsin(2 * q0 * q3 + 2 * q1 * q2)
        
        # np.clip protects against float inaccuracies
        inner_term = np.clip(2 * w * z + 2 * x * y, -1.0, 1.0)
        beta_0_hat = np.arcsin(inner_term)
        
        # COST: Minimize variance of the angle on the constrained axis
        cost = np.var(beta_0_hat)
        return cost

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
            
            if is_flat_valley:
                delta_w = self.delta_w_minus_1
                opt_duration = time.time() - opt_start
                optimization_success = True
                cost_fun_val = 0.0
            else:
                # --- 3-STAGE COARSE-TO-FINE GRID SEARCH (SciPy Replacement) ---
                
                # 1. GROBE SUCHE (36 Punkte, 10° Auflösung)
                coarse_space = np.linspace(-np.pi, np.pi, 36)
                best_coarse_cost = float('inf')
                best_coarse_yaw = 0.0
                
                for yaw in coarse_space:
                    cost = self._cost_function(yaw, r_parent_inv, r_child)
                    if cost < best_coarse_cost:
                        best_coarse_cost = cost
                        best_coarse_yaw = yaw
                        
                # 2. FEINE SUCHE (21 Punkte, 1° Auflösung im Radius von +/- 10° um den groben Treffer)
                fine_radius = np.deg2rad(10.0)
                fine_space = np.linspace(best_coarse_yaw - fine_radius, best_coarse_yaw + fine_radius, 21)
                best_fine_cost = best_coarse_cost
                best_fine_yaw = best_coarse_yaw
                
                for yaw in fine_space:
                    cost = self._cost_function(yaw, r_parent_inv, r_child)
                    if cost < best_fine_cost:
                        best_fine_cost = cost
                        best_fine_yaw = yaw

                # 3. ULTRA-FEINE SUCHE (21 Punkte, 0.1° Auflösung im Radius von +/- 1° um den feinen Treffer)
                ultra_fine_radius = np.deg2rad(1.0)
                ultra_fine_space = np.linspace(best_fine_yaw - ultra_fine_radius, best_fine_yaw + ultra_fine_radius, 21)
                best_cost = best_fine_cost
                best_yaw = best_fine_yaw

                for yaw in ultra_fine_space:
                    cost = self._cost_function(yaw, r_parent_inv, r_child)
                    if cost < best_cost:
                        best_cost = cost
                        best_yaw = yaw
                
                opt_duration = time.time() - opt_start
                optimization_success = True
                delta_w = best_yaw
                cost_fun_val = best_cost
            
            if optimization_success:
                self.w_index += 1
                w_idx = self.w_index
                
                # --- HEADING FILTER / BIAS EXTRAPOLATION ---
                if self.enable_singularity_filter:
                    is_singular = (r_w < self.r_min) or is_flat_valley
                    s_w = 0.0 if is_singular else r_w
                    if is_flat_valley:
                        print(f"💤 [2D] Flat Valley ACTIVE! Arm held still. Using bias extrapolation.")
                    elif is_singular:
                        print(f"⚠️ [2D] Singularity filter ACTIVE! (Rating (r_w): {r_w:.3f} < {self.r_min}). Using bias extrapolation.")
                else:
                    is_singular = False
                    s_w = 1.0
                
                # Calculate adaptive factors
                k_b_w =     max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                # Calculate new drift rate and filtered offset
                b_w = self.b_w_minus_1 + s_w * k_b_w * (delta_w - self.delta_w_minus_1 - self.b_w_minus_1)
                delta_f_w = self.delta_f_w_minus_1 + b_w + s_w * k_delta_w * (delta_w - self.delta_f_w_minus_1 - b_w)
                
                self.b_w_minus_1 = b_w
                self.delta_w_minus_1 = delta_w
                self.delta_f_w_minus_1 = delta_f_w
                
                self.target_heading_offset = delta_f_w
                self.current_heading_offset = self.target_heading_offset
                print(f"[{time.strftime('%H:%M:%S')}] 2D Optimizer (Shoulder): Target offset = {np.degrees(self.current_heading_offset):.2f}° (r_w: {r_w:.2f}; duration: {opt_duration:.3f}s)")
                
                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"{time.time()},{w_idx},{r_w:.4f},{int(is_singular)},{np.degrees(delta_w):.4f},{np.degrees(b_w):.4f},{np.degrees(delta_f_w):.4f},{cost_fun_val:.6f}\n")
            else:
                print("⚠️ 2D optimization failed for this window.")
        except Exception as e:
            print(f"Error in 2D Optimizer-Thread: {e}")
        finally:
            self.is_calculating = False