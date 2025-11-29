import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class R2MatrixGenerator:
    def __init__(self, df, target_column):
        """
        初始化R²矩阵生成器
        
        参数:
        df: 包含所有变量的DataFrame
        target_column: 目标变量列名
        """
        self.df = df.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]
        self.scaler = StandardScaler()
        self.results = {}
        
    def get_all_feature_combinations(self, max_features=None):
        """
        获取所有可能的特征组合
        
        参数:
        max_features: 最大特征数量，None表示使用所有特征
        """
        if max_features is None:
            max_features = len(self.feature_columns)
        
        all_combinations = []
        for k in range(1, max_features + 1):
            combinations = list(itertools.combinations(self.feature_columns, k))
            all_combinations.extend(combinations)
        
        return all_combinations
    
    def calculate_r2_for_combination(self, feature_combination):
        """
        计算特定特征组合的R²值
        """
        try:
            # 提取特征和目标
            X = self.df[list(feature_combination)].values
            y = self.df[self.target_column].values
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练线性回归模型
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # 预测并计算R²
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            # 获取系数
            coefficients = model.coef_
            intercept = model.intercept_
            
            return r2, coefficients, intercept
        
        except Exception as e:
            print(f"计算组合 {feature_combination} 时出错: {e}")
            return 0, [], 0
    
    def generate_r2_matrix(self, max_features=None, sort_by_r2=True):
        """
        生成R²矩阵
        
        参数:
        max_features: 最大特征数量
        sort_by_r2: 是否按R²值排序
        """
        # 获取所有特征组合
        combinations = self.get_all_feature_combinations(max_features)
        
        print(f"共有 {len(combinations)} 种特征组合需要计算")
        
        # 计算每种组合的R²值
        results = []
        for i, combo in enumerate(combinations):
            if i % 50 == 0:  # 每50个组合打印一次进度
                print(f"计算进度: {i+1}/{len(combinations)}")
                
            r2, coefficients, intercept = self.calculate_r2_for_combination(combo)
            
            results.append({
                'combination': combo,
                'features': list(combo),
                'num_features': len(combo),
                'r2': r2,
                'coefficients': coefficients,
                'intercept': intercept
            })
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 按R²值排序
        if sort_by_r2:
            results_df = results_df.sort_values('r2', ascending=False)
        
        # 重置索引
        results_df = results_df.reset_index(drop=True)
        
        self.results_df = results_df
        return results_df
    
    def create_r2_matrix_heatmap(self, top_n=20):
        """
        创建R²值热力图矩阵
        
        参数:
        top_n: 显示前N个最佳组合
        """
        if not hasattr(self, 'results_df'):
            print("请先运行 generate_r2_matrix()")
            return
        
        # 选择前N个最佳组合
        top_results = self.results_df.head(top_n).copy()
        
        # 创建矩阵数据
        matrix_data = []
        feature_set = set()
        
        for _, row in top_results.iterrows():
            features = row['features']
            r2 = row['r2']
            feature_set.update(features)
            
            # 为每个特征组合创建一行
            row_data = {'combination': ', '.join(features), 'r2': r2}
            for feature in feature_set:
                row_data[feature] = 1 if feature in features else 0
            matrix_data.append(row_data)
        
        # 创建矩阵DataFrame
        matrix_df = pd.DataFrame(matrix_data)
        
        # 设置组合为索引
        matrix_df = matrix_df.set_index('combination')
        
        # 只保留特征列和R²列
        feature_cols = list(feature_set)
        matrix_df = matrix_df[feature_cols + ['r2']]
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                       gridspec_kw={'height_ratios': [1, 4]})
        
        # 上子图：R²值条形图
        r2_values = matrix_df['r2'].values
        colors = plt.cm.viridis((r2_values - r2_values.min()) / (r2_values.max() - r2_values.min()))
        ax1.bar(range(len(r2_values)), r2_values, color=colors)
        ax1.set_xticks(range(len(r2_values)))
        ax1.set_xticklabels(matrix_df.index, rotation=45, ha='right')
        ax1.set_ylabel('R²值')
        ax1.set_title('不同特征组合的R²值')
        
        # 下子图：特征组合热力图
        feature_matrix = matrix_df[feature_cols].T
        sns.heatmap(feature_matrix, annot=True, cmap='Blues', 
                   cbar_kws={'label': '特征存在 (1=是, 0=否)'},
                   ax=ax2)
        ax2.set_ylabel('特征')
        ax2.set_xlabel('特征组合')
        ax2.set_title('特征组合矩阵')
        
        plt.tight_layout()
        plt.show()
        
        return matrix_df
    
    def get_best_combinations(self, n=10):
        """
        获取前N个最佳特征组合
        """
        if not hasattr(self, 'results_df'):
            print("请先运行 generate_r2_matrix()")
            return
        
        return self.results_df.head(n)[['features', 'r2', 'num_features']]
    
    def find_optimal_combination(self, r2_threshold=0.9, max_features=None):
        """
        寻找满足R²阈值的最简单特征组合
        
        参数:
        r2_threshold: R²阈值
        max_features: 最大特征数量
        """
        if not hasattr(self, 'results_df'):
            print("请先运行 generate_r2_matrix()")
            return
        
        # 筛选满足R²阈值的组合
        valid_combinations = self.results_df[self.results_df['r2'] >= r2_threshold]
        
        if valid_combinations.empty:
            print(f"没有找到R² >= {r2_threshold}的组合")
            return None
        
        # 按特征数量排序，选择最简单的组合
        simplest_combination = valid_combinations.sort_values('num_features').iloc[0]
        
        print(f"满足R² >= {r2_threshold}的最简单组合:")
        print(f"特征: {simplest_combination['features']}")
        print(f"R²值: {simplest_combination['r2']:.4f}")
        print(f"特征数量: {simplest_combination['num_features']}")
        
        return simplest_combination

# 使用示例
def main():
    # 创建示例数据（替换为您的实际数据）
    ExcelFile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")
    df=pd.read_excel(ExcelFile,skiprows=0,usecols="B:F",sheet_name="Sheet6")
    df.columns=["总收入","游客数量","交通指数","环境指数","海南省A级景区数"]
    
    # 生成示例数据
    data = {
        '游客数量': np.array(df["游客数量"].to_list()),
        '交通指数': np.array(df["交通指数"].to_list()),
        '环境指数': np.array(df["环境指数"].to_list()),
        '海南省A级景区数': np.array(df["海南省A级景区数"].to_list()),
        '总收入':np.array(df["总收入"].to_list())
    }
    
    df = pd.DataFrame(data)
    
    # 创建R²矩阵生成器
    r2_generator = R2MatrixGenerator(df, '总收入')
    
    # 生成R²矩阵（限制最大特征数为4以加快计算）
    print("开始计算R²矩阵...")
    r2_matrix = r2_generator.generate_r2_matrix(max_features=4)
    
    # 显示前10个最佳组合
    print("\n前10个最佳特征组合:")
    best_combinations = r2_generator.get_best_combinations(10)
    for i, (_, row) in enumerate(best_combinations.iterrows()):
        print(f"{i+1}. 特征: {row['features']}, R²: {row['r2']:.4f}, 特征数: {row['num_features']}")
    
    # 创建热力图矩阵
    print("\n生成热力图矩阵...")
    heatmap_matrix = r2_generator.create_r2_matrix_heatmap(top_n=15)
    
    # 寻找最优组合
    print("\n寻找最优特征组合...")
    optimal = r2_generator.find_optimal_combination(r2_threshold=0.8)
    
    # 保存结果到Excel
    r2_matrix.to_excel('r2_matrix_results.xlsx', index=False)
    print("\n结果已保存到 'r2_matrix_results.xlsx'")
    
    return r2_generator, r2_matrix

# 针对您的实际数据的使用方法
def use_with_your_data():
    """
    使用您自己的数据
    """
    # 读取您的数据
    # df = pd.read_excel("您的数据文件.xlsx")
    
    # 假设您的DataFrame已经加载，目标变量是'总收入'
    # r2_generator = R2MatrixGenerator(df, '总收入')
    
    # 生成R²矩阵
    # r2_matrix = r2_generator.generate_r2_matrix(max_features=5)  # 限制最大特征数
    
    # 显示结果
    # print(r2_matrix.head(10))
    
    pass

if __name__ == "__main__":
    r2_generator, r2_matrix = main()