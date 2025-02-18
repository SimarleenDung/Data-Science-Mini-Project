import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


file_path = "C:/Users/asus/Desktop/question_one_data.xlsx"

data = pd.read_excel(file_path, sheet_name="Sheet1")
data_sorted = data.sort_values(by="Age_Ma", ascending=True).reset_index(drop=True)

# 将数据分为5组
n_groups = 10
data_sorted["Age_Group"], bins = pd.qcut(data_sorted["Age_Ma"], q=n_groups, labels=False, retbins=True)
# 显示每个年龄组的区间
print("年龄组区间：")
for i in range(n_groups):
    print(f"Group {i}: {bins[i]:.2f} - {bins[i+1]:.2f} Ma")
# 分组
groups = data_sorted.groupby("Age_Group")

# 通过 BIC 选择最优模型
def find_best_gmm(data, max_components=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    bic_values = []
    models = []
    for n in range(2, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(scaled_data)
        bic_values.append(gmm.bic(scaled_data))
        models.append(gmm)
    best_parameter = np.argmin(bic_values) + 1
    best_model = models[np.argmin(bic_values)]
    return best_parameter, best_model, bic_values

# 对每个年龄组训练
results = {}
for group_id, group_data in groups:
    features = group_data[["Max_Diameter_µm", "Elongation", "Min_Diameter_µm", "Shape Factor", "Sphericity"]]
    best_parameter, best_model, bic_values = find_best_gmm(features, max_components=10)
    results[group_id] = {
        "best_parameter": best_parameter,
        "best_model": best_model,
        "bic_values": bic_values
    }

# 可视化 BIC 曲线,用来确定每个年龄组内最佳的物种数量，BIC越小越好
for group_id, result in results.items():
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(result["bic_values"]) + 1), result["bic_values"], marker="o", linestyle="-")
    plt.xlabel("Number of Components")
    plt.ylabel("Value")
    plt.title(f"Group {group_id}: BIC vs Number of Components")
    plt.show()

# 提取平均年龄和最优物种数
age_means = groups["Age_Ma"].mean()
species_counts = {group_id: result["best_parameter"] for group_id, result in results.items()}
# 转换为 DataFrame 并排序
species_df = pd.DataFrame({
    "Age_Group": species_counts.keys(),
    "Species_Count": species_counts.values(),
    "Mean_Age": [age_means[group_id] for group_id in species_counts.keys()]
}).sort_values(by="Mean_Age")

# 可视化物种数量变化
plt.figure(figsize=(12, 6))
plt.plot(species_df["Mean_Age"], species_df["Species_Count"], marker="o", linestyle="-", color="r")
plt.xlabel("Mean Age (Ma)")
plt.ylabel("Number of Species")
plt.title("Species Number Over Time ")
plt.grid(True)
plt.show()