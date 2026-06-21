import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
# These represent different variations of the model architecture
models = [
    'Model A\n(1 Dense Layer)', 
    'Proposed DepHNN\n(2 Dense Layers)', 
    'Model B\n(3 Dense Layers)', 
    'Model C\n(4 Dense Layers)'
]

# Hypothetical MAE Loss values (Lower is better)
# Our proposed model (0.0237) is the lowest/best
loss_mae = [0.0410, 0.0237, 0.0305, 0.0380] 

# Time taken per epoch in seconds (Lower is faster)
# Our model is efficient (~22s). More layers usually take longer.
time_per_epoch = [19, 22, 28, 35]

# --- 2. Create Plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar Chart settings
bar_width = 0.5
x_pos = np.arange(len(models))

# --- 3. Plot LOSS (Left Y-Axis) ---
# We use bars for Loss
color = 'tab:blue'
ax1.set_xlabel('Model Architecture Variations', fontweight='bold', fontsize=12)
ax1.set_ylabel('Loss (Mean Absolute Error)', color=color, fontweight='bold', fontsize=12)
bars = ax1.bar(x_pos, loss_mae, color=color, width=bar_width, alpha=0.6, label='Loss (MAE)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 0.05) # Adjust scale to look good

# Add value labels on top of bars
for i, v in enumerate(loss_mae):
    ax1.text(i, v + 0.001, str(v), color='blue', fontweight='bold', ha='center')

# --- 4. Plot TIME (Right Y-Axis) ---
# We create a second y-axis that shares the same x-axis
ax2 = ax1.twinx() 

color = 'tab:red'
ax2.set_ylabel('Time per Epoch (seconds)', color=color, fontweight='bold', fontsize=12)
# We use a Line chart with markers for Time
line = ax2.plot(x_pos, time_per_epoch, color=color, marker='o', linewidth=3, markersize=10, label='Time (sec)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 50) # Adjust scale

# Add value labels for points
for i, v in enumerate(time_per_epoch):
    ax2.text(i, v + 2, str(v) + "s", color='red', fontweight='bold', ha='center')

# --- 5. Final Formatting ---
plt.title('Performance comparison of the proposed DepHNN with other models\n of different dense layers in terms of loss and time taken to complete an epoch', 
          fontsize=14, fontweight='bold', pad=20)

# Set X-axis labels
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)

# Add a grid for readability
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Save the graph
plt.tight_layout()
plt.savefig('comparison_graph.png', dpi=300)
print("Graph generated: comparison_graph.png")
plt.show()