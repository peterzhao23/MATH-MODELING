import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler,MinMaxScaler #数据处理模块
np.set_printoptions(precision=5,suppress=True)

#创建预处理函数
class Critic:
     def preproc(df, cost_columns,benefit_columns):
          X=df.values.astype(float)#将X转化为np数组，转化为浮点数，若为整数，最后结果只有0，1
          xmin=X.min(axis=0)
          xmax=X.max(axis=0)
          xmaxmin=xmax-xmin
          m,n=X.shape
          for i in range(m):
               for j in range(n):
                    if df.columns[j] in cost_columns:
                         X[i,j] = float((xmax[j] - X[i,j]) / xmaxmin[j])
                    elif df.columns[j] in benefit_columns:
                         X[i,j] =  float((X[i,j] - xmin[j]) / xmaxmin[j])
          return X



     def conflict(X_standard):  #计算标准差（对比强度）
         return np.std(X_standard,axis=0,ddof=1)
     


     def relate(df,X_standard): #计算变量相关性与冲突性
          m,n=X_standard.shape
          copy=pd.DataFrame(X_standard,columns=df.columns)
          cometrix=copy.corr()
          f=[0.0]*n
          co_values=cometrix.values
          for j in range(n):
               for i in range(n):
                    f[j]=1-np.abs(co_values[i,j])+f[j]
               f[j]=float(f[j])
          return f

def critic(df,cost_columns,benefit_columns):
     X_standard=Critic.preproc(df,cost_columns,benefit_columns) #得到归一化矩阵
     R=Critic.conflict(X_standard) 
     f=np.array(Critic.relate(df,X_standard))   #相关性
     information_content=R*f  #计算信息量
     weight=np.array([information_content[i]/sum(information_content) for i in range(len(df.columns))]) #计算权重
     return weight


#topsis排序法
def topsis(X_standard,weight_row,lenth):
#构造加权规范矩阵
     weight_matrix=np.array(weight_row).reshape(1,-1) #将权重转换成矩阵，reshape保留原列表行与列（1表示行，-1表示列）
     X_weight=X_standard*weight_matrix
           
#确定正理想解与负理想解
     Vmin=X_weight.min(axis=0)
     Vmax=X_weight.max(axis=0)
                              
        
#计算每个评价对象到理想解的距离
     dist_best = np.sqrt(np.sum((X_weight - Vmax) ** 2, axis=1))
     dist_worst = np.sqrt(np.sum((X_weight - Vmin) ** 2, axis=1))

     #得到相对贴近度
     #T=np.array([float(dist_worst[i])/float((dist_best[i]+dist_worst[i])) for i in range(lenth)])
     T=dist_worst/(dist_best+dist_worst)
     return T

#熵权法,忽略变量相关性
def entropy(df,cost_columns,benefit_columns):
#数据归一化，平移    
    Y=Critic.preproc(df,cost_columns,benefit_columns)
    
#计算比重
    Y_sum=np.sum(Y,axis=0)
    divweight=Y/Y_sum
#计算熵值 
    k=1/math.log(len(df))
    e=-k*(np.sum((divweight+1e-10)*np.log(divweight+1e-10),axis=0))
#计算差异系数
    g=1-e
#计算权重
    weight=g/np.sum(g)
    return  weight


#采用Z-score标准化
def standardlize(X):
    V=StandardScaler()
    X=V.fit_transform(X)
    return X
#采用MinMax归一化
def normalize(X):
    V=MinMaxScaler()
    if X.ndim==1:
        X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

    else:
        X=V.fit_transform(X)
    return X


                          
        


