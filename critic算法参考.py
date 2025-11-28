import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 1. 模拟环境数据 (5个监测点位，4个环境指标)
# 假设指标：PM2.5(μg/m³, 成本型), SO2(μg/m³, 成本型), 水质优良率(%, 效益型), 噪声(dB, 成本型)
data = {
    'PM2.5': [35, 42, 28, 50, 38],
    'SO2': [10, 15, 8, 20, 12],
    'Water_Quality': [80, 65, 90, 55, 75],
    'Noise': [55, 60, 50, 65, 58]
}
df = pd.DataFrame(data)
print("原始环境数据:")
print(df)

# 2. 数据标准化
def standardize_matrix(df, benefit_columns=None, cost_columns=None):
    """
    对数据框进行标准化处理
    benefit_columns: 效益型指标列名列表
    cost_columns: 成本型指标列名列表
    """
    if benefit_columns is None:
        benefit_columns = []
    if cost_columns is None:
        cost_columns = []
    
    X = df.values.copy()
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xmaxmin = xmax - xmin
    
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            if df.columns[j] in cost_columns:
                # 成本型指标：越小越好，逆向处理
                X[i, j] = (xmax[j] - X[i, j]) / xmaxmin[j]
            elif df.columns[j] in benefit_columns:
                # 效益型指标：越大越好，正向处理
                X[i, j] = (X[i, j] - xmin[j]) / xmaxmin[j]
    return X

# 指定效益型和成本型指标
benefit_cols = ['Water_Quality']  # 水质优良率是效益型
cost_cols = ['PM2.5', 'SO2', 'Noise']  # 其他是成本型

X_normalized = standardize_matrix(df, benefit_columns=benefit_cols, cost_columns=cost_cols)
print("\n标准化后的矩阵:")
print(X_normalized)

# 3. 计算对比强度 (标准差)
std_devs = np.std(X_normalized, axis=0, ddof=1)  # 注意设置ddof=1[citation:6]
print("\n各指标的标准差 (对比强度):")
for i, col in enumerate(df.columns):
    print(f"{col}: {std_devs[i]:.4f}")

# 4. 计算冲突性 (基于相关系数)
n_features = X_normalized.shape[1]
corr_matrix = np.corrcoef(X_normalized, rowvar=False)
conflict = np.sum(1 - np.abs(corr_matrix), axis=1)
print("\n指标间的相关系数矩阵:")
print(corr_matrix)
print("\n各指标的冲突性:")
for i, col in enumerate(df.columns):
    print(f"{col}: {conflict[i]:.4f}")

# 5. 计算信息量
information_content = std_devs * conflict
print("\n各指标的信息量:")
for i, col in enumerate(df.columns):
    print(f"{col}: {information_content[i]:.4f}")

# 6. 计算权重
weights = information_content / np.sum(information_content)
print("\n各指标的最终权重:")
for i, col in enumerate(df.columns):
    print(f"{col}: {weights[i]:.4f}")

# 7. 计算环境指数并排序
environmental_index = np.dot(X_normalized, weights)
df['Environmental_Index'] = environmental_index
df_sorted = df.sort_values('Environmental_Index', ascending=False)
print("\n各监测点位的环境指数及排名:")
print(df_sorted)