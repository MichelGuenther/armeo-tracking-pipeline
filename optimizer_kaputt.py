import numpy as np
from tracking_modes import ModeKinematicConstraintsStrategy1D, ModeClassicStrategy1D, ModeDualSeedRefereeStrategy1D, ModeKinematicConstraintsStrategy2D, ModeClassicStrategy2D, ModeDualSeedRefereeStrategy2D
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
    def __init__(self, sensor_upper, sensor_lower, window_size=200, step_size=100, flat_valley_threshold=1e-4, enable_singularity_filter=True, enable_flat_valley_filter=True, enable_anti_windup=True, enable_valley_retry_validation=True, tau_b_=2.8, tau_delta_=0.7, delta_delta_weight=0.0, limrom_mode="off", mode_pure_constraint_referee=False, log_file="drift_log_1D.csv", debug_log_file=None):
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
        self.mode_pure_constraint_referee = mode_pure_constraint_referee
        self.limrom_mode = limrom_mode
        if self.limrom_mode == "dual_seed_referee":
            self.mode_pure_constraint_referee = True
            
        self.rom_violation_counter = 0  # Hysteresis Counter
        self.delta_delta_weight = delta_delta_weight
        self.log_file = log_file
        self.debug_log_file = debug_log_file
        
        self.latest_angle_x = 0.0

        if self.limrom_mode == "kinematic_constraints":
            self.mode_strategy = ModeKinematicConstraintsStrategy1D(self)
        elif self.limrom_mode == "classic":
            self.mode_strategy = ModeClassicStrategy1D(self)
        elif self.limrom_mode == "dual_seed_referee":
            self.mode_strategy = ModeDualSeedRefereeStrategy1D(self)
        else:
            raise ValueError(f"Unknown mode: {self.limrom_mode}")
        
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
        return self.current_heading_offset, self.latest_angle_x

    def _run_optimization_threaded(self, buf_up, buf_low):
        self.is_calculating = True
        try:
            quat_diffs_up = np.diff(buf_up, axis=0)
            quat_diffs_low = np.diff(buf_low, axis=0)
            movement_var_up = np.sum(quat_diffs_up**2)
            movement_var_low = np.sum(quat_diffs_low**2)
            
            is_flat_valley = (movement_var_up < self.flat_valley_threshold and movement_var_low < self.flat_valley_threshold)
            print(f"🔍 [1D] Variance Check | Up: {movement_var_up:.6E}, Low: {movement_var_low:.6E} | Threshold: {self.flat_valley_threshold:.6E}")
            
            r_upper = R.from_quat(buf_up)
            r_lower = R.from_quat(buf_low)
            r_upper_inv = r_upper.inv()
            
            z_comp = r_upper.apply([1, 0, 0])[:, 2]
            r_w_k = np.sqrt(np.clip(1.0 - z_comp**2, 0.0, 1.0))
            r_w = np.sqrt(np.mean(r_w_k**2))
            
            opt_start = time.time()
            
            is_singular_val = (r_w < self.r_min)
            should_bypass_opt = (is_flat_valley and self.enable_flat_valley_filter) or \
                                (is_singular_val and self.enable_singularity_filter)

            if should_bypass_opt:
                delta_w = self.delta_w_minus_1
                opt_duration = time.time() - opt_start
                optimization_success = True
                cost_fun_val = 0.0
            else:
                initial_guess = [self.delta_f_w_minus_1]
                
                self.latest_jump_event = False
                self.latest_seed_lost = False
                
                optimization_success, best_yaw, best_cost = self.mode_strategy.run(r_upper_inv, r_lower, initial_guess, movement_var_up, movement_var_low, is_flat_valley, r_w)
                
                if hasattr(self, '_current_debug_info'):
                    del self._current_debug_info
                    
                opt_duration = time.time() - opt_start
                delta_w = best_yaw
                cost_fun_val = best_cost
            
            if optimization_success:
                self.w_index += 1
                w_idx = self.w_index
                
                if self.enable_singularity_filter:
                    is_singular_due_to_flat_valley = is_flat_valley and self.enable_flat_valley_filter
                    is_singular = (r_w < self.r_min) or is_singular_due_to_flat_valley
                    s_w = 0.0 if is_singular else r_w
                    if is_singular_due_to_flat_valley:
                        print(f"💤 [1D] Flat Valley ACTIVE! Arm held still. Using bias extrapolation.")
                    elif is_singular:
                        print(f"⚠️ [1D] Singularity filter ACTIVE! (Rating (r_w): {r_w:.3f} < {self.r_min}). Using bias extrapolation.")
                   
                    if is_singular and self.enable_anti_windup:
                        delta_w = self.delta_f_w_minus_1 + self.b_w_minus_1
                else:
                    is_singular = False
                    s_w = 1.0
                
                # (Der Jump Override wurde hier entfernt, da er den initial_guess im Off-Mode korrumpiert)
                        
                k_b_w =     max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                b_w = self.b_w_minus_1 + s_w * k_b_w * (delta_w - self.delta_w_minus_1 - self.b_w_minus_1)
                delta_f_w = self.delta_f_w_minus_1 + b_w + s_w * k_delta_w * (delta_w - self.delta_f_w_minus_1 - b_w)
                
                self.b_w_minus_1 = b_w
                self.delta_w_minus_1 = delta_w
                self.delta_f_w_minus_1 = delta_f_w
                
                self.target_heading_offset = delta_f_w
                self.current_heading_offset = self.target_heading_offset
                
                q_up = R.from_quat(buf_up[-1]).inv()
                q_low_corrected = R.from_euler('z', delta_f_w, degrees=False) * R.from_quat(buf_low[-1])
                q_rel = q_up * q_low_corrected
                avg_angles = q_rel.as_euler('XYZ', degrees=True)
                
                quats = q_rel.as_quat()
                if quats.ndim == 1:
                    x, y, z, w = quats[0], quats[1], quats[2], quats[3]
                else:
                    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

                angle_x_rad = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
                angle_y_rad = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
                angle_z_rad = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))

                angle_x_deg = np.rad2deg(angle_x_rad)
                angle_y_deg = np.rad2deg(angle_y_rad)
                angle_z_deg = np.rad2deg(angle_z_rad)
                
                self.latest_angle_x = angle_x_deg
                
                print(f"[{w_idx}] 1D Opt: Δw={np.degrees(delta_w):5.1f}° | Δf,w={np.degrees(delta_f_w):5.1f}° | b_w={np.degrees(b_w):5.2f}°/w | s_w={s_w:.2f} | 1D-Euler (x={angle_x_deg:5.1f}°, y={angle_y_deg:5.1f}°, z={angle_z_deg:5.1f}°)")

            self.is_calculating = False
            return self.target_heading_offset

        except Exception as e:
            print(f"Error in 1D thread: {e}")
            import traceback
            traceback.print_exc()
            self.is_calculating = False
            return 0.0

class Optimizer2D_Universal:
    """
    This optimizer implements the 2D joint constraints for a universal joint (e.g., simplified shoulder).
    A 2-DoF joint allows rotation around two axes (e.g., flexion/extension and abduction/adduction), 
    but blocks the third axis (e.g., the axial internal/external rotation along the bone).
    The optimizer searches for the heading correction angle (delta_yaw) that minimizes the variance 
    on this single "forbidden" axis.
    """
    def __init__(self, sensor_parent, sensor_child, window_size=200, step_size=100, flat_valley_threshold=1e-8, enable_singularity_filter=True, enable_flat_valley_filter=True, enable_anti_windup=True, enable_valley_retry_validation=True, enable_limrom=False, limrom_mode="off", mode_pure_constraint_referee=False, tau_b_=2.8, tau_delta_=0.7, delta_delta_weight=0.0, log_file="drift_log_2D.csv", debug_log_file=None):
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
        self.mode_pure_constraint_referee = False
        if self.limrom_mode == "dual_seed_referee":
            self.mode_pure_constraint_referee = True
            
        self.latest_angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        if self.limrom_mode == "kinematic_constraints":
            self.mode_strategy = ModeKinematicConstraintsStrategy2D(self)
        elif self.limrom_mode == "classic":
            self.mode_strategy = ModeClassicStrategy2D(self)
        elif self.limrom_mode == "dual_seed_referee":
            self.mode_strategy = ModeDualSeedRefereeStrategy2D(self)
        else:
            raise ValueError(f"Unknown mode: {self.limrom_mode}")
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
        return self.current_heading_offset, self.latest_angles

    def _run_optimization_threaded(self, buf_par, buf_chi):
        self.is_calculating = True
        try:
            quat_diffs_par = np.diff(buf_par, axis=0)
            quat_diffs_chi = np.diff(buf_chi, axis=0)
            movement_var_par = np.sum(quat_diffs_par**2)
            movement_var_chi = np.sum(quat_diffs_chi**2)

            is_flat_valley = (movement_var_par < self.flat_valley_threshold and movement_var_chi < self.flat_valley_threshold)
            print(f"🔍 [2D] Variance Check | Parent: {movement_var_par:.6E}, Child: {movement_var_chi:.6E} | Threshold: {self.flat_valley_threshold:.6E}")

            r_parent = R.from_quat(buf_par)
            r_child = R.from_quat(buf_chi)
            r_parent_inv = r_parent.inv()

            j1_z = r_parent.apply([0, 1, 0])[:, 2]
            j2_z = r_child.apply([1, 0, 0])[:, 2]

            proj_j1 = np.sqrt(np.clip(1.0 - j1_z**2, 0.0, 1.0))
            proj_j2 = np.sqrt(np.clip(1.0 - j2_z**2, 0.0, 1.0))
            r_w_k = proj_j1 * proj_j2
            r_w = np.sqrt(np.mean(r_w_k**2))

            opt_start = time.time()

            is_singular_val = (r_w < self.r_min)
            should_bypass_opt = (is_flat_valley and self.enable_flat_valley_filter) or \
                                (is_singular_val and self.enable_singularity_filter)

            if should_bypass_opt:
                delta_w = self.delta_w_minus_1
                opt_duration = time.time() - opt_start
                optimization_success = True
                cost_fun_val = 0.0
            else:
                initial_guess = [self.delta_f_w_minus_1]

                self.latest_jump_event = False
                self.latest_seed_lost = False

                optimization_success, best_yaw, best_cost = self.mode_strategy.run(r_parent_inv, r_child, initial_guess, movement_var_par, movement_var_chi, is_flat_valley, r_w)
                
                if hasattr(self, '_current_debug_info'):
                    del self._current_debug_info

                opt_duration = time.time() - opt_start
                delta_w = best_yaw
                cost_fun_val = best_cost
            
            if optimization_success:
                self.w_index += 1
                w_idx = self.w_index
                
                if self.enable_singularity_filter:
                    is_singular_due_to_flat_valley = is_flat_valley and self.enable_flat_valley_filter
                    is_singular = (r_w < self.r_min) or is_singular_due_to_flat_valley
                    s_w = 0.0 if is_singular else r_w
                    if is_singular_due_to_flat_valley:
                        print(f"💤 [2D] Flat Valley ACTIVE! Arm held still. Using bias extrapolation.")
                    elif is_singular:
                        print(f"⚠️ [2D] Singularity filter ACTIVE! (Rating (r_w): {r_w:.3f} < {self.r_min}). Using bias extrapolation.")
                   
                    if is_singular and self.enable_anti_windup:
                        delta_w = self.delta_f_w_minus_1 + self.b_w_minus_1
                else:
                    is_singular = False
                    s_w = 1.0
                
                # (Der Jump Override wurde hier entfernt, da er den initial_guess im Off-Mode korrumpiert)
                        

                k_b_w =     max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_b), 1.0 / w_idx)
                k_delta_w = max(1.0 - np.exp(-np.log(2) * self.T_s / self.tau_delta), 1.0 / w_idx)
                
                b_w = self.b_w_minus_1 + s_w * k_b_w * (delta_w - self.delta_w_minus_1 - self.b_w_minus_1)
                delta_f_w = self.delta_f_w_minus_1 + b_w + s_w * k_delta_w * (delta_w - self.delta_f_w_minus_1 - b_w)
                
                self.b_w_minus_1 = b_w
                self.delta_w_minus_1 = delta_w
                self.delta_f_w_minus_1 = delta_f_w
                
                self.target_heading_offset = delta_f_w
                self.current_heading_offset = self.target_heading_offset
                
                rot_offset_best = R.from_euler('z', delta_f_w, degrees=False)
                r_rel_best = r_parent_inv * (rot_offset_best * r_child)
                avg_angles = np.mean(r_rel_best.as_euler('XYZ', degrees=True), axis=0)
                quats = r_rel_best.as_quat()

                x = quats[:, 0]
                y = quats[:, 1]
                z = quats[:, 2]
                w = quats[:, 3]

                angle_x_rad = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
                angle_y_rad = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
                angle_z_rad = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

                angle_x_deg = np.rad2deg(angle_x_rad)
                angle_y_deg = np.rad2deg(angle_y_rad)
                angle_z_deg = np.rad2deg(angle_z_rad)
                
                self.latest_angles['x'] = np.mean(angle_x_deg)
                self.latest_angles['y'] = np.mean(angle_y_deg)
                self.latest_angles['z'] = np.mean(angle_z_deg)

            self.is_calculating = False
            return self.target_heading_offset

        except Exception as e:
            print(f"Error in 2D thread: {e}")
            import traceback
            traceback.print_exc()
            self.is_calculating = False
            return 0.0

