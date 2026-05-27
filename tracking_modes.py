
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ==========================================================
# 1D TRACKING STRATEGIEN
# ==========================================================
class TrackerStrategy1D:
    def __init__(self, opt):
        self.opt = opt
        
    def log_coarse_search(self, r_upper_inv, r_lower, residual_func):
        if not self.opt.debug_log_file or not hasattr(self.opt, '_current_debug_info'):
            return
            
        info = self.opt._current_debug_info
        for test_deg in range(-180, 180, 5):
            test_rad = np.deg2rad(test_deg)
            res_coarse = residual_func([test_rad], r_upper_inv, r_lower)
            cost_coarse = np.sum(res_coarse**2)
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{info['w_index']},{test_deg:.4f},{cost_coarse:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")

    def run(self, r_upper_inv, r_lower, initial_guess, movement_var_up, movement_var_low, is_flat_valley, r_w):
        raise NotImplementedError


class ModeKinematicConstraintsStrategy1D(TrackerStrategy1D):
    def residuals(self, params, r_upper_inv, r_lower):
        delta_yaw = params[0]
        rot_offset = R.from_euler('x', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_lower
        rel_rot = r_upper_inv * r_child_corrected
        
        q_scipy = rel_rot.as_quat()
        x, y, z, w = q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2], q_scipy[..., 3]
        beta_0_hat = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        if self.opt.delta_delta_weight > 0.0:
            reg = (self.opt.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.opt.delta_delta_weight)
            res_list.append([reg])
            
        return np.concatenate(res_list)

    def run(self, r_upper_inv, r_lower, initial_guess, movement_var_up, movement_var_low, is_flat_valley, r_w):
        self.log_coarse_search(r_upper_inv, r_lower, self.residuals)
        
        if self.opt.w_index == 0:
            best_grid_cost = float('inf')
            best_grid_yaw = initial_guess[0]
            for test_deg in range(-180, 180, 5):
                test_rad = np.deg2rad(test_deg)
                res_val = self.residuals([test_rad], r_upper_inv, r_lower)
                c = np.sum(res_val**2)
                if c < best_grid_cost:
                    best_grid_cost = c
                    best_grid_yaw = test_rad
            res = least_squares(self.residuals, [best_grid_yaw], args=(r_upper_inv, r_lower), method='lm')
        else:
            res = least_squares(self.residuals, initial_guess, args=(r_upper_inv, r_lower), method='lm')
            
        best_yaw = res.x[0]
        best_cost = res.cost * 2.0
        
        if self.opt.debug_log_file:
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.opt.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
        return True, best_yaw, best_cost


class ModeClassicStrategy1D(TrackerStrategy1D):
    def residuals_with_limrom(self, params, r_upper_inv, r_lower):
        delta_yaw = params[0]
        rot_offset = R.from_euler('x', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_lower
        rel_rot = r_upper_inv * r_child_corrected
        
        q_scipy = rel_rot.as_quat()
        x, y, z, w = q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2], q_scipy[..., 3]
        beta_0_hat = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        angle_z = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        z_lower_bound = np.deg2rad(-20.0)
        z_upper_bound = np.deg2rad(150.0)
        
        penalty_over_z = np.maximum(0, angle_z - z_upper_bound)
        penalty_under_z = np.maximum(0, z_lower_bound - angle_z)
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        if getattr(self.opt, 'limrom_mode', 'classic') == "paper":
            if (angle_z > z_upper_bound).any() or (angle_z < z_lower_bound).any(): 
                res_list.append(np.array([10.0]))
        else:
            res_list.extend([penalty_over_z, penalty_under_z])
            
        if self.opt.delta_delta_weight > 0.0:
            reg = (self.opt.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.opt.delta_delta_weight)
            res_list.append([reg])
            
        return np.concatenate(res_list)

    def residuals_kinematics_only(self, params, r_upper_inv, r_lower):
        delta_yaw = params[0]
        rot_offset = R.from_euler('x', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_lower
        rel_rot = r_upper_inv * r_child_corrected
        
        q_scipy = rel_rot.as_quat()
        x, y, z, w = q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2], q_scipy[..., 3]
        beta_0_hat = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        if self.opt.delta_delta_weight > 0.0:
            reg = (self.opt.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.opt.delta_delta_weight)
            res_list.append([reg])
            
        return np.concatenate(res_list)

    def run(self, r_upper_inv, r_lower, initial_guess, movement_var_up, movement_var_low, is_flat_valley, r_w):
        if self.opt.w_index == 0:
            self.log_coarse_search(r_upper_inv, r_lower, self.residuals_with_limrom)
            
            best_grid_cost = float('inf')
            best_grid_yaw = initial_guess[0]
            for test_deg in range(-180, 180, 5):
                test_rad = np.deg2rad(test_deg)
                res_val = self.residuals_with_limrom([test_rad], r_upper_inv, r_lower)
                c = np.sum(res_val**2)
                if c < best_grid_cost:
                    best_grid_cost = c
                    best_grid_yaw = test_rad
            
            res = least_squares(self.residuals_with_limrom, [best_grid_yaw], args=(r_upper_inv, r_lower), method='lm')
        else:
            self.log_coarse_search(r_upper_inv, r_lower, self.residuals_kinematics_only)
            res = least_squares(self.residuals_kinematics_only, initial_guess, args=(r_upper_inv, r_lower), method='lm')
            
        best_yaw = res.x[0]
        best_cost = res.cost * 2.0
        
        if self.opt.debug_log_file:
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.opt.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
        return True, best_yaw, best_cost


class ModeDualSeedRefereeStrategy1D(ModeClassicStrategy1D):
    def run(self, r_upper_inv, r_lower, initial_guess, movement_var_up, movement_var_low, is_flat_valley, r_w):
        self.log_coarse_search(r_upper_inv, r_lower, self.residuals_kinematics_only)
        
        seed_a = self.opt.delta_f_w_minus_1
        seed_b = self.opt.seed_B_w_minus_1
        
        res_a = least_squares(self.residuals_kinematics_only, [seed_a], args=(r_upper_inv, r_lower), method='lm')
        res_b = least_squares(self.residuals_kinematics_only, [seed_b], args=(r_upper_inv, r_lower), method='lm')
        
        sol_a = res_a.x[0]
        sol_b = res_b.x[0]
        
        sol_diff = np.abs((sol_a - sol_b + np.pi) % (2*np.pi) - np.pi)
        if sol_diff < np.deg2rad(10.0):
            self.opt.seed_lost_counter += 1
            if self.opt.seed_lost_counter > 5:
                sol_b = sol_a + np.pi
                res_b = least_squares(self.residuals_kinematics_only, [sol_b], args=(r_upper_inv, r_lower), method='lm')
                sol_b = res_b.x[0]
                self.opt.latest_seed_lost = True
        else:
            self.opt.seed_lost_counter = 0

        self.opt.seed_B_w_minus_1 = sol_b
        
        cost_a_limrom = np.sum(self.residuals_with_limrom([sol_a], r_upper_inv, r_lower)**2)
        cost_b_limrom = np.sum(self.residuals_with_limrom([sol_b], r_upper_inv, r_lower)**2)
        
        if cost_a_limrom <= cost_b_limrom + 0.1:
            self.opt.rom_violation_counter = max(0, self.opt.rom_violation_counter - 1)
        else:
            self.opt.rom_violation_counter += 1
            
        if self.opt.rom_violation_counter >= 3:
            best_yaw = sol_b
            best_cost = cost_b_limrom
            
            self.opt.b_w_minus_1 = 0.0
            self.opt.delta_f_w_minus_1 = sol_b
            self.opt.delta_w_minus_1 = sol_b
            
            self.opt.seed_B_w_minus_1 = sol_a
            self.opt.rom_violation_counter = 0
            self.opt.latest_jump_event = True
            print(f"⚖️ [1D] REFEREE ENTSCHEIDUNG: Seed B gewinnt! Harter Flip ausgeführt.")
        else:
            best_yaw = sol_a
            best_cost = cost_a_limrom
            
        if self.opt.debug_log_file:
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.opt.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_up:.6E},{movement_var_low:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
        return True, best_yaw, best_cost

# ==========================================================
# 2D TRACKING STRATEGIEN
# ==========================================================
class TrackerStrategy2D:
    def __init__(self, opt):
        self.opt = opt
        
    def log_coarse_search(self, r_parent_inv, r_child, residual_func):
        if not self.opt.debug_log_file or not hasattr(self.opt, '_current_debug_info'):
            return
            
        info = self.opt._current_debug_info
        for test_deg in range(-180, 180, 5):
            test_rad = np.deg2rad(test_deg)
            res_coarse = residual_func([test_rad], r_parent_inv, r_child)
            cost_coarse = np.sum(res_coarse**2)
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{info['w_index']},{test_deg:.4f},{cost_coarse:.6f},0,{info['var_up']:.6E},{info['var_low']:.6E},{info['is_flat']},{info['r_w']:.4f},0.0\n")

    def run(self, r_parent_inv, r_child, initial_guess, movement_var_par, movement_var_chi, is_flat_valley, r_w):
        raise NotImplementedError


class ModeKinematicConstraintsStrategy2D(TrackerStrategy2D):
    def residuals(self, params, r_parent_inv, r_child):
        delta_yaw = params[0]
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_child
        rel_rot = r_parent_inv * r_child_corrected
        
        q_scipy = rel_rot.as_quat()
        x, y, z, w = q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2], q_scipy[..., 3]
        beta_0_hat = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        if self.opt.delta_delta_weight > 0.0:
            reg = (self.opt.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.opt.delta_delta_weight)
            res_list.append([reg])
            
        return np.concatenate(res_list)

    def run(self, r_parent_inv, r_child, initial_guess, movement_var_par, movement_var_chi, is_flat_valley, r_w):
        self.log_coarse_search(r_parent_inv, r_child, self.residuals)
        
        if self.opt.w_index == 0:
            best_grid_cost = float('inf')
            best_grid_yaw = initial_guess[0]
            for test_deg in range(-180, 180, 5):
                test_rad = np.deg2rad(test_deg)
                res_val = self.residuals([test_rad], r_parent_inv, r_child)
                c = np.sum(res_val**2)
                if c < best_grid_cost:
                    best_grid_cost = c
                    best_grid_yaw = test_rad
            res = least_squares(self.residuals, [best_grid_yaw], args=(r_parent_inv, r_child), method='lm')
        else:
            res = least_squares(self.residuals, initial_guess, args=(r_parent_inv, r_child), method='lm')
            
        best_yaw = res.x[0]
        best_cost = res.cost * 2.0
        
        if self.opt.debug_log_file:
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.opt.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_par:.6E},{movement_var_chi:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
        return True, best_yaw, best_cost


class ModeClassicStrategy2D(TrackerStrategy2D):
    def residuals_with_limrom(self, params, r_parent_inv, r_child):
        delta_yaw = params[0]
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_child
        rel_rot = r_parent_inv * r_child_corrected
        
        q_scipy = rel_rot.as_quat()
        x, y, z, w = q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2], q_scipy[..., 3]
        beta_0_hat = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
        
        x_lower_bound = np.deg2rad(-30.0)
        x_upper_bound = np.deg2rad(30.0)
        y_lower_bound = np.deg2rad(-40.0)
        y_upper_bound = np.deg2rad(60.0)
        
        penalty_over_x = np.maximum(0, angle_x - x_upper_bound)
        penalty_under_x = np.maximum(0, x_lower_bound - angle_x)
        penalty_over_y = np.maximum(0, angle_y - y_upper_bound)
        penalty_under_y = np.maximum(0, y_lower_bound - angle_y)
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        if getattr(self.opt, 'limrom_mode', 'classic') == "paper":
            if (angle_x > x_upper_bound).any() or (angle_x < x_lower_bound).any() or (angle_y > y_upper_bound).any() or (angle_y < y_lower_bound).any(): 
                res_list.append(np.array([10.0]))
        else:
            res_list.extend([penalty_over_x, penalty_under_x, penalty_over_y, penalty_under_y])
            
        if self.opt.delta_delta_weight > 0.0:
            reg = (self.opt.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.opt.delta_delta_weight)
            res_list.append([reg])
            
        return np.concatenate(res_list)

    def residuals_kinematics_only(self, params, r_parent_inv, r_child):
        delta_yaw = params[0]
        rot_offset = R.from_euler('z', delta_yaw, degrees=False)
        r_child_corrected = rot_offset * r_child
        rel_rot = r_parent_inv * r_child_corrected
        
        q_scipy = rel_rot.as_quat()
        x, y, z, w = q_scipy[..., 0], q_scipy[..., 1], q_scipy[..., 2], q_scipy[..., 3]
        beta_0_hat = np.arcsin(np.clip(2 * w * z + 2 * x * y, -1.0, 1.0))
        
        res_list = [np.atleast_1d(beta_0_hat)]
        
        if self.opt.delta_delta_weight > 0.0:
            reg = (self.opt.delta_f_w_minus_1 - delta_yaw) * np.sqrt(self.opt.delta_delta_weight)
            res_list.append([reg])
            
        return np.concatenate(res_list)

    def run(self, r_parent_inv, r_child, initial_guess, movement_var_par, movement_var_chi, is_flat_valley, r_w):
        if self.opt.w_index == 0:
            self.log_coarse_search(r_parent_inv, r_child, self.residuals_with_limrom)
            best_grid_cost = float('inf')
            best_grid_yaw = initial_guess[0]
            for test_deg in range(-180, 180, 5):
                test_rad = np.deg2rad(test_deg)
                res_val = self.residuals_with_limrom([test_rad], r_parent_inv, r_child)
                c = np.sum(res_val**2)
                if c < best_grid_cost:
                    best_grid_cost = c
                    best_grid_yaw = test_rad
            
            res = least_squares(self.residuals_with_limrom, [best_grid_yaw], args=(r_parent_inv, r_child), method='lm')
        else:
            self.log_coarse_search(r_parent_inv, r_child, self.residuals_kinematics_only)
            res = least_squares(self.residuals_kinematics_only, initial_guess, args=(r_parent_inv, r_child), method='lm')
            
        best_yaw = res.x[0]
        best_cost = res.cost * 2.0
        
        if self.opt.debug_log_file:
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.opt.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_par:.6E},{movement_var_chi:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
        return True, best_yaw, best_cost


class ModeDualSeedRefereeStrategy2D(ModeClassicStrategy2D):
    def run(self, r_parent_inv, r_child, initial_guess, movement_var_par, movement_var_chi, is_flat_valley, r_w):
        self.log_coarse_search(r_parent_inv, r_child, self.residuals_kinematics_only)
        seed_a = self.opt.delta_f_w_minus_1
        seed_b = self.opt.seed_B_w_minus_1
        
        res_a = least_squares(self.residuals_kinematics_only, [seed_a], args=(r_parent_inv, r_child), method='lm')
        res_b = least_squares(self.residuals_kinematics_only, [seed_b], args=(r_parent_inv, r_child), method='lm')
        
        sol_a = res_a.x[0]
        sol_b = res_b.x[0]
        
        sol_diff = np.abs((sol_a - sol_b + np.pi) % (2*np.pi) - np.pi)
        if sol_diff < np.deg2rad(10.0):
            self.opt.seed_lost_counter += 1
            if self.opt.seed_lost_counter > 5:
                sol_b = sol_a + np.pi
                res_b = least_squares(self.residuals_kinematics_only, [sol_b], args=(r_parent_inv, r_child), method='lm')
                sol_b = res_b.x[0]
                self.opt.latest_seed_lost = True
        else:
            self.opt.seed_lost_counter = 0

        self.opt.seed_B_w_minus_1 = sol_b
        
        cost_a_limrom = np.sum(self.residuals_with_limrom([sol_a], r_parent_inv, r_child)**2)
        cost_b_limrom = np.sum(self.residuals_with_limrom([sol_b], r_parent_inv, r_child)**2)
        
        if cost_a_limrom <= cost_b_limrom + 0.1:
            self.opt.rom_violation_counter = max(0, self.opt.rom_violation_counter - 1)
        else:
            self.opt.rom_violation_counter += 1
            
        if self.opt.rom_violation_counter >= 3:
            best_yaw = sol_b
            best_cost = cost_b_limrom
            
            self.opt.b_w_minus_1 = 0.0
            self.opt.delta_f_w_minus_1 = sol_b
            self.opt.delta_w_minus_1 = sol_b
            
            self.opt.seed_B_w_minus_1 = sol_a
            self.opt.rom_violation_counter = 0
            self.opt.latest_jump_event = True
            print(f"⚖️ [2D] REFEREE ENTSCHEIDUNG: Seed B gewinnt! Harter Flip ausgeführt.")
        else:
            best_yaw = sol_a
            best_cost = cost_a_limrom
            
        if self.opt.debug_log_file:
            with open(self.opt.debug_log_file, "a") as f:
                f.write(f"{time.time()},{self.opt.w_index + 1},{np.degrees(best_yaw):.4f},{best_cost:.6f},1,{movement_var_par:.6E},{movement_var_chi:.6E},{int(is_flat_valley)},{r_w:.4f},{np.degrees(best_yaw):.4f}\n")
                
        return True, best_yaw, best_cost
