import os

file_path = "optimizer.py"
with open(file_path, "r") as f:
    content = f.read()

# 1D Patch
old_res_1d = "    def _residuals(self, delta_yaw_array, r_upper_inv, r_lower_window):"
new_res_1d = """    def _eval_limrom_cost(self, delta_yaw, r_upper_inv, r_lower_window):
        orig_limrom = self.limrom_mode
        self.limrom_mode = "classic"
        orig_weight = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        res = self._residuals([delta_yaw], r_upper_inv, r_lower_window)
        cost = np.sum(res**2)
        self.limrom_mode = orig_limrom
        self.delta_delta_weight = orig_weight
        return cost

    def _residuals(self, delta_yaw_array, r_upper_inv, r_lower_window):"""
if "def _eval_limrom_cost(self, delta_yaw, r_upper_inv" not in content:
    content = content.replace(old_res_1d, new_res_1d)

# 2D Patch
old_res_2d = "    def _residuals(self, delta_yaw_array, r_parent_inv, r_child_window):"
new_res_2d = """    def _eval_limrom_cost(self, delta_yaw, r_parent_inv, r_child_window):
        orig_limrom = self.limrom_mode
        self.limrom_mode = "classic"
        orig_weight = self.delta_delta_weight
        self.delta_delta_weight = 0.0
        res = self._residuals([delta_yaw], r_parent_inv, r_child_window)
        cost = np.sum(res**2)
        self.limrom_mode = orig_limrom
        self.delta_delta_weight = orig_weight
        return cost

    def _residuals(self, delta_yaw_array, r_parent_inv, r_child_window):"""
if "def _eval_limrom_cost(self, delta_yaw, r_parent_inv" not in content:
    content = content.replace(old_res_2d, new_res_2d)

with open(file_path, "w") as f:
    f.write(content)
print("Methods patched!")
