from sklearn.linear_model import LinearRegression   #scikit-learn中线性回归模型
from sklearn.datasets import make_regression  #生成回归数据集的工具函数
from sklearn.model_selection import train_test_split #用于将数据集分割为训练集和测试集
from sklearn.metrics import r2_score,mean_squared_error  #模型评估指标：判定系数 标准差
from sklearn.preprocessing import MinMaxScaler #数据处理模块
import pandas as pd                                 #处理数据
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np

#数据可视化初始化 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ExcelFile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")
df=pd.read_excel(ExcelFile,skiprows=0,usecols="B:F",sheet_name="Sheet6")
df.columns=["总收入","游客数量","交通指数","环境指数","海南省A级景区数"]


class tourismmodel:
    #初始化模型   
    def __init__(self):
        self.model=LinearRegression()
        self.is_fitted=False
        self.scaler=MinMaxScaler()
    #用变量相关性矩阵分析各个变量的线性关系，简化高度相关的变量
    def exporatory_data_analysis(self,df):
        correlation_matrix=df.corr() #计算皮尔逊相关系数矩阵
        print(correlation_matrix.round(3)) #输出矩阵

        #矩阵可视化
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title(f'{df.columns.to_list()}间相关性热力图')
        plt.tight_layout()
        plt.savefig('热力图',bbox_inches="tight",dpi=128)


    def modeling(self,df):
         #数据准备
         X=df[["游客数量","交通指数","环境指数","海南省A级景区数"]]
         Y=df['总收入']

         #自变量归一化
         X[["游客数量","海南省A级景区数"]]=self.scaler.fit_transform(X[["游客数量","海南省A级景区数"]])  #归一化自变量
       
         
        
        #按照时间顺序数排列数据
         #df=df.sort_values["年份"]
         
         self.model.fit(X,Y)
         self.is_fitted=True
          # 预测
         Y_pred = self.model.predict(X)
        
        # 评估模型
         r2 = r2_score(Y, Y_pred)
         mse = mean_squared_error(Y, Y_pred)
         rmse = np.sqrt(mse)
         ra2=1-(1-r2)*(len(df)-1)/(len(df)-len(X.columns)-1) #复判定系数削弱多自变量的影响
         print(f"\n=== 模型评估结果 ===")
         print(f"截距 (β₀): {self.model.intercept_:.2f}")
         print("回归系数:")
         for var, coef in zip(X.columns, self.model.coef_):
            print(f"  {var} (β): {coef:.6f}")
        
         print(f"\n模型性能指标:")
         print(f"R² Score: {r2:.4f}")
         print(f"MSE: {mse:.2f}")
         print(f"RMSE: {rmse:.2f}")
         print(f"Ra²: {ra2:.2f}")
test=tourismmodel()
test.modeling(df)
