import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

file_path = "C:/Users/asus/Desktop/question_one_data.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")
data_sorted = data.sort_values(by="Age_Ma", ascending=True).reset_index(drop=True)

size_of_age_piece = 0.5
step = 0.4
max_components = 10
min_components = 2
min_samples = 20

time_points = np.arange(
    data_sorted["Age_Ma"].min(),
    data_sorted["Age_Ma"].max() - size_of_age_piece + step,
    step
)

get_results = []
for t in time_points:
    time_start = t
    time_end = t + size_of_age_piece

    window_data = data_sorted[
        (data_sorted["Age_Ma"] >= time_start) &
        (data_sorted["Age_Ma"] < time_end)
        ]


    features = window_data[["Max_Diameter_µm", "Elongation", "Min_Diameter_µm", "Shape Factor", "Sphericity"]]
    Q1 = features.quantile(0.25)
    Q3 = features.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = window_data[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]
    features = filtered_data[["Max_Diameter_µm", "Elongation", "Min_Diameter_µm", "Shape Factor", "Sphericity"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    bic_values = []
    for n in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(scaled_features)
        bic_values.append(gmm.bic(scaled_features))

    best_n = np.argmin(bic_values) + min_components

    # 使用最佳聚类数重新拟合GMM
    gmm = GaussianMixture(n_components=best_n, random_state=42)
    gmm.fit(scaled_features)
    labels = gmm.predict(scaled_features)  # 获取聚类标签

    # 计算轮廓系数
    silhouette_avg = silhouette_score(scaled_features, labels)

    get_results.append({
        "time_start": time_start,
        "time_end": time_end,
        "Mean_Age": (time_start + time_end) / 2,
        "Best_N_Components": best_n,
        "BIC_Values": bic_values,
        "Silhouette_Score": silhouette_avg,
    })

    plt.figure(figsize=(8, 5))
    plt.plot(range(min_components, max_components + 1), bic_values, marker="o", linestyle="-", color="darkgreen")
    plt.axvline(best_n, color="red", linestyle="--", label=f"Best N = {best_n}")
    plt.xlabel("Number of Components", fontsize=12)
    plt.ylabel("BIC Value", fontsize=12)
    plt.title(f"Window {time_start:.2f}-{time_end:.2f} Ma\nOptimal Clusters: {best_n}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


results_df = pd.DataFrame(get_results)

results_df["Time_Window"] = results_df.apply(
    lambda row: f"{row['time_start']:.2f}-{row['time_end']:.2f} Ma",
    axis=1
)

# 绘制轮廓系数随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(results_df["Mean_Age"], results_df["Silhouette_Score"], marker="o", linestyle="-", color="purple")
plt.xlabel("Age (Ma)", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.title("Silhouette Score Over Time", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 8))
ax = plt.gca()
ax.plot(
    results_df["Mean_Age"],
    results_df["Best_N_Components"],
    marker="o",
    markersize=10,
    markerfacecolor="white",
    markeredgecolor="darkblue",
    markeredgewidth=2,
    linestyle="-",
    color="darkblue",
    linewidth=2,
    label="Species Count"
)

# 标注时间和物种数
for idx, row in results_df.iterrows():
    ax.text(
        row["Mean_Age"],
        row["Best_N_Components"] + 0.15,
        f"{row['Best_N_Components']}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="darkred",
        fontweight="bold"
    )
    ax.text(
        row["Mean_Age"],
        row["Best_N_Components"] - 0.15,
        f"({row['Time_Window']})",
        ha="center",
        va="top",
        fontsize=8,
        color="grey",
        rotation=45
    )


ax.set_xticks(results_df["Mean_Age"])
ax.set_xticklabels(results_df["Time_Window"], rotation=45, ha="right", fontsize=10)
ax.set_xlabel("Age ma", fontsize=14, labelpad=15)
ax.set_ylabel("Number of Species", fontsize=14, labelpad=10)
ax.set_title("Change in Species Number ", fontsize=16, pad=20)
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()