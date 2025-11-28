import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 设置中文字体（如果图表需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TourismRevenueModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_and_prepare_data(self, data_path=None):
        """
        加载和准备数据
        如果data_path为None，则生成模拟数据用于演示
        """
        if data_path:
            # 从文件加载真实数据
            df = pd.read_csv(data_path)
        else:
            # 生成模拟数据用于演示
            np.random.seed(42)
            n_samples = 200
            
            # 生成自变量（设置一些相关性以模拟真实情况）
            T = np.random.normal(100000, 30000, n_samples)  # 游客数量
            R = np.random.normal(500, 150, n_samples)       # 人均消费
            V = np.random.normal(7, 2, n_samples)           # 交通指数
            E = np.random.normal(8, 1.5, n_samples)         # 环境指数
            D = np.random.normal(75, 15, n_samples)         # 旅游业发展指数
            
            # 生成因变量（基于理论关系加上随机噪声）
            # M = 10000 + 0.1*T + 80*R + 5000*V + 3000*E + 200*D + 噪声
            noise = np.random.normal(0, 100000, n_samples)
            M = 10000 + 0.1*T + 80*R + 5000*V + 3000*E + 200*D + noise
            
            # 创建DataFrame
            df = pd.DataFrame({
                'T': T, 'R': R, 'V': V, 'E': E, 'D': D, 'M': M
            })
            
            # 确保没有负值
            df = df.abs()
        
        return df
    
    def exploratory_data_analysis(self, df):
        """探索性数据分析"""
        print("=== 探索性数据分析 ===")
        print(f"数据集形状: {df.shape}")
        print("\n数据前5行:")
        print(df.head())
        
        print("\n基本统计描述:")
        print(df.describe())
        
        print("\n缺失值检查:")
        print(df.isnull().sum())
        
        # 相关性分析
        print("\n变量间相关性矩阵:")
        correlation_matrix = df.corr()
        print(correlation_matrix.round(3))
        
        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('变量间相关性热力图')
        plt.tight_layout()
        plt.show()
        
        # 因变量与各自变量的散点图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        independent_vars = ['T', 'R', 'V', 'E', 'D']
        for i, var in enumerate(independent_vars):
            axes[i].scatter(df[var], df['M'], alpha=0.6)
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('M (旅游业总收入)')
            axes[i].set_title(f'M vs {var}')
            
            # 添加趋势线
            z = np.polyfit(df[var], df['M'], 1)
            p = np.poly1d(z)
            axes[i].plot(df[var], p(df[var]), "r--", alpha=0.8)
        
        # 隐藏多余的子图
        for i in range(len(independent_vars), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def check_multicollinearity(self, X):
        """检查多重共线性"""
        print("\n=== 多重共线性诊断 ===")
        
        # 计算VIF（方差膨胀因子）
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        
        print("方差膨胀因子(VIF):")
        print(vif_data)
        
        # VIF判断标准：>10表示严重多重共线性
        high_vif = vif_data[vif_data["VIF"] > 10]
        if len(high_vif) > 0:
            print(f"\n警告：以下变量存在严重多重共线性:")
            print(high_vif)
        else:
            print("\n多重共线性在可接受范围内")
            
        return vif_data
    
    def build_model(self, df, test_size=0.2, random_state=42, standardize=False):
        """建立和训练多元线性回归模型"""
        print("\n=== 建立多元线性回归模型 ===")
        
        # 准备数据
        X = df[['T', 'R', 'V', 'E', 'D']]  # 自变量
        y = df['M']  # 因变量
        
        print(f"自变量: {list(X.columns)}")
        print(f"因变量: M (旅游业总收入)")
        
        # 检查多重共线性
        self.check_multicollinearity(X)
        
        # 数据标准化（可选）
        if standardize:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            print("已对自变量进行标准化处理")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集样本数: {X_train.shape[0]}")
        print(f"测试集样本数: {X_test.shape[0]}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 评估模型
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\n=== 模型评估结果 ===")
        print(f"截距 (β₀): {self.model.intercept_:.2f}")
        print("回归系数:")
        for var, coef in zip(X.columns, self.model.coef_):
            print(f"  {var} (β): {coef:.6f}")
        
        print(f"\n模型性能指标:")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        # 使用statsmodels进行详细统计检验
        X_with_const = sm.add_constant(X_train)
        model_sm = sm.OLS(y_train, X_with_const).fit()
        
        print(f"\n=== 详细统计检验结果 ===")
        print(model_sm.summary())
        
        return X_train, X_test, y_train, y_test, y_pred, model_sm
    
    def analyze_residuals(self, y_test, y_pred):
        """残差分析"""
        if not self.is_fitted:
            print("请先训练模型")
            return
        
        residuals = y_test - y_pred
        
        plt.figure(figsize=(15, 10))
        
        # 1. 残差散点图
        plt.subplot(2, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        
        # 2. 残差Q-Q图（正态性检验）
        plt.subplot(2, 3, 2)
        sm.qqplot(residuals, line='45', fit=True)
        plt.title('Q-Q图（残差正态性检验）')
        
        # 3. 残差直方图
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('残差')
        plt.ylabel('频数')
        plt.title('残差分布')
        
        # 4. 实际值vs预测值
        plt.subplot(2, 3, 4)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('实际值 vs 预测值')
        
        # 5. 残差自相关图
        plt.subplot(2, 3, 5)
        pd.Series(residuals).plot(alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('样本序号')
        plt.ylabel('残差')
        plt.title('残差序列图')
        
        plt.tight_layout()
        plt.show()
        
        # 残差统计检验
        from scipy import stats
        print("\n=== 残差统计分析 ===")
        print(f"残差均值: {residuals.mean():.2f}")
        print(f"残差标准差: {residuals.std():.2f}")
        
        # 正态性检验（Shapiro-Wilk检验）
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"残差正态性检验 p-value: {shapiro_p:.4f}")
        if shapiro_p > 0.05:
            print("残差服从正态分布（p > 0.05）")
        else:
            print("残差不服从正态分布（p ≤ 0.05）")
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        if not self.is_fitted:
            print("请先训练模型")
            return
        
        # 基于标准化系数的特征重要性
        importance = pd.DataFrame({
            'feature': ['T', 'R', 'V', 'E', 'D'],
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n=== 特征重要性分析 ===")
        print("按系数绝对值排序:")
        print(importance[['feature', 'coefficient']])
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in importance['coefficient']]
        plt.barh(importance['feature'], importance['coefficient'], color=colors)
        plt.xlabel('回归系数')
        plt.title('特征重要性（回归系数）')
        plt.axvline(x=0, color='black', linestyle='-')
        plt.tight_layout()
        plt.show()
        
        return importance
    
    def predict_new_data(self, new_data):
        """对新数据进行预测"""
        if not self.is_fitted:
            print("请先训练模型")
            return
        
        predictions = self.model.predict(new_data)
        return predictions

# 使用示例
def main():
    # 创建模型实例
    tourism_model = TourismRevenueModel()
    
    # 加载数据（这里使用模拟数据）
    df = tourism_model.load_and_prepare_data()
    
    # 探索性数据分析
    tourism_model.exploratory_data_analysis(df)
    
    # 建立模型
    X_train, X_test, y_train, y_test, y_pred, model_sm = tourism_model.build_model(df)
    
    # 残差分析
    tourism_model.analyze_residuals(y_test, y_pred)
    
    # 特征重要性分析
    importance = tourism_model.feature_importance_analysis()
    
    # 模型解释和应用建议
    print("\n=== 模型解释和应用建议 ===")
    print("1. 经济意义解释:")
    print("   - 游客数量(T)的系数：表示每增加1名游客，旅游总收入的变化")
    print("   - 人均消费(R)的系数：表示人均消费每增加1元，旅游总收入的变化") 
    print("   - 交通指数(V)的系数：表示交通便利度对旅游收入的边际贡献")
    print("   - 环境指数(E)的系数：表示环境质量对旅游收入的边际贡献")
    print("   - 发展指数(D)的系数：表示整体发展水平对旅游收入的边际贡献")
    
    print("\n2. 政策建议方向:")
    print("   - 重点关注系数较大的变量，这些是影响旅游总收入的关键因素")
    print("   - 根据系数的正负和大小，制定相应的旅游业发展策略")
    print("   - 考虑变量间的协同效应，制定综合发展计划")

if __name__ == "__main__":
    main()