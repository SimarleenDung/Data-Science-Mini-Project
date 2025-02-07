# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset (Ensure df_q1data_fixed is already cleaned)
df_q1data_fixed = pd.read_excel("q1_data")  # Uncomment if loading from a file

# Set plot style
sns.set(style="whitegrid")

# Create a figure for subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# 📌 1. Scatter Plot: Max Diameter vs. Age (Million Years)
sns.scatterplot(
    data=df_q1data_fixed,
    x="Age_Ma",
    y="Max_Diameter_µm",
    hue="Age_Group",
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title("Scatter Plot: Max Diameter vs. Age")
axes[0].set_xlabel("Age (Ma)")
axes[0].set_ylabel("Max Diameter (µm)")

# 📌 2. Boxplot: Distribution of Max Diameter over Age Groups
sns.boxplot(
    data=df_q1data_fixed,
    x="Age_Group",
    y="Max_Diameter_µm",
    ax=axes[1]
)
axes[1].set_title("Box Plot: Max Diameter Distribution Across Age Groups")
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("Max Diameter (µm)")

# 📌 3. Line Plot: Mean Max Diameter Over Time
mean_size_over_time = df_q1data_fixed.groupby("Age_Ma")["Max_Diameter_µm"].mean().reset_index()
sns.lineplot(
    data=mean_size_over_time,
    x="Age_Ma",
    y="Max_Diameter_µm",
    marker="o",
    ax=axes[2]
)
axes[2].set_title("Line Plot: Mean Max Diameter Over Time")
axes[2].set_xlabel("Age (Ma)")
axes[2].set_ylabel("Mean Max Diameter (µm)")

# Adjust layout and show plots
plt.tight_layout()
plt.show()
