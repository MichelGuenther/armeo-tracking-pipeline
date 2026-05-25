import os
import re

file_path = "optimizer.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Füge Variablen im 2D __init__ hinzu
old_init_2d = """        self.latest_angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}"""
new_init_2d = """        self.latest_angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.latest_jump_event = False
        self.latest_seed_lost = False"""
content = content.replace(old_init_2d, new_init_2d)

# 2. Füge _eval_limrom_cost zu Optimizer1D (vor _residuals) hinzu
if "def _eval_limrom_cost(self, delta_yaw, r_upper_inv, r_lower):" not in content:
    old_res_1d = "    def _residuals(self, delta_yaw, r_upper_inv, r_lower):"
    new_res_1d = """    def _eval_limrom_cost(self, delta_yaw, r_upper_inv, r_lower):
        orig_limrom = self.limrom_mode
        self.limrom_mode = "classic"
        orig_weight = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        res = self._residuals([delta_yaw], r_upper_inv, r_lower)
        cost = np.sum(res**2)
        self.limrom_mode = orig_limrom
        self.delta_delta_weight = orig_weight
        return cost

    def _residuals(self, delta_yaw, r_upper_inv, r_lower):"""
    content = content.replace(old_res_1d, new_res_1d)

# 3. Füge _eval_limrom_cost zu Optimizer2D_Universal (vor _residuals) hinzu
if "def _eval_limrom_cost(self, delta_yaw, r_parent_inv, r_child):" not in content:
    old_res_2d = "    def _residuals(self, delta_yaw, r_parent_inv, r_child):"
    new_res_2d = """    def _eval_limrom_cost(self, delta_yaw, r_parent_inv, r_child):
        orig_limrom = self.limrom_mode
        self.limrom_mode = "classic"
        orig_weight = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        res = self._residuals([delta_yaw], r_parent_inv, r_child)
        cost = np.sum(res**2)
        self.limrom_mode = orig_limrom
        self.delta_delta_weight = orig_weight
        return cost

    def _residuals(self, delta_yaw, r_parent_inv, r_child):"""
    content = content.replace(old_res_2d, new_res_2d)

with open(file_path, "w") as f:
    f.write(content)
print("Fixes applied successfully!")
