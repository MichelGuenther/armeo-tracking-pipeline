import numpy as np
from scipy.spatial.transform import Rotation as R

# Simuliere einen Arm, der von 0 bis 120 Grad Abduktion (Y) bewegt wird,
# ohne jegliche Torsion (Z = 0) und ohne Flexion (X = 0).
# In SXYZ ist Y die zweite Achse.
y_angles = np.linspace(0, 120, 13)

print("Y_True | w, x, y, z | Angle_X_Euler | Angle_Y_Euler | Angle_Z_Euler | Twist_Z")

for y_deg in y_angles:
    r = R.from_euler('xyz', [0, y_deg, 0], degrees=True)
    q = r.as_quat()
    w = q[3]
    x = q[0]
    y = q[1]
    z = q[2]
    
    # Alte Euler Methode
    angle_x = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    angle_y = np.arcsin(np.clip(2 * w * y - 2 * z * x, -1.0, 1.0))
    angle_z = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Neue Swing-Twist Methode (Twist um Z)
    # Normierung für Twist: sqrt(w^2 + z^2)
    norm = np.sqrt(w**2 + z**2)
    if norm > 1e-6:
        twist_z = 2.0 * np.arctan2(z, w)
    else:
        twist_z = 0.0
        
    print(f"{y_deg:6.1f} | {w:4.2f} {x:4.2f} {y:4.2f} {z:4.2f} | "
          f"{np.degrees(angle_x):6.1f} | {np.degrees(angle_y):6.1f} | {np.degrees(angle_z):6.1f} | {np.degrees(twist_z):6.1f}")
