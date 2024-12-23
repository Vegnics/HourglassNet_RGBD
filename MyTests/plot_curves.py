import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file into a pandas DataFrame
file_path = "/home/quinoa/Desktop/some_shit/patient_project/HourglassNet_RGBD/logs/myModelLogs.csv"#'/home/quinoa/Desktop/myModelLogs.csv'  # Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot PCKh@0.5 and val_PCKh@0.5 on the same axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('PCKh@0.5', color='tab:blue')
ax1.plot(df['epoch'], df['PCKh@0.5'], label='PCKh@0.5', color='tab:blue', marker='o')
ax1.plot(df['epoch'], df['val_PCKh@0.5'], label='Val PCKh@0.5', color='tab:cyan', linestyle='--', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for the loss and val_loss
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:red')
ax2.plot(df['epoch'], df['loss'], label='Loss', color='tab:red', marker='x')
ax2.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:orange', linestyle='--', marker='x')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add legends
fig.tight_layout()  # Adjust layout to prevent overlap
fig.legend(loc='upper center', bbox_to_anchor=(0.9, 0.9),fontsize=15)

# Show the plot
plt.title('Training Metrics: PCKh@0.5, Loss, Val PCKh@0.5, Val Loss')
plt.show()