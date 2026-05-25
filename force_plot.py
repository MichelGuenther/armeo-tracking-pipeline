import os

plot_path = "/home/michel/Desktop/02_Programmierung_ubuntu/armeo-tracking-pipeline-main/plot_debug_lm.py"
with open(plot_path, "r") as f:
    content = f.read()

# Replace the specific block in the detail plot
old_block = """    if len(seed_a) > 0:
        ax.scatter(seed_a['tested_yaw_deg'], seed_a['cost_val'], color='gold', s=150, marker='v', edgecolor='black', linewidth=1, label='Seed A (History)', zorder=8)
    if len(seed_b) > 0:
        ax.scatter(seed_b['tested_yaw_deg'], seed_b['cost_val'], color='orange', s=150, marker='^', edgecolor='black', linewidth=1, label='Seed B (Mirrored)', zorder=8)"""

new_block = """    if len(seed_a) > 0:
        yaw = seed_a['tested_yaw_deg'].values[0]
        cost = seed_a['cost_val'].values[0]
        lr_cost = seed_a['best_yaw_deg'].values[0]  # Misused column for LimRom
        ax.scatter(yaw, cost, color='gold', s=150, marker='v', edgecolor='black', linewidth=1, label='Seed A (History)', zorder=8)
        ax.annotate(f"LimRom: {lr_cost:.1f}", (yaw, cost), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='darkgoldenrod')
    if len(seed_b) > 0:
        yaw = seed_b['tested_yaw_deg'].values[0]
        cost = seed_b['cost_val'].values[0]
        lr_cost = seed_b['best_yaw_deg'].values[0]  # Misused column for LimRom
        ax.scatter(yaw, cost, color='orange', s=150, marker='^', edgecolor='black', linewidth=1, label='Seed B (Mirrored)', zorder=8)
        ax.annotate(f"LimRom: {lr_cost:.1f}", (yaw, cost), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='orangered')"""

content = content.replace(old_block, new_block)

with open(plot_path, "w") as f:
    f.write(content)

print("Plot script fully updated with annotations!")
