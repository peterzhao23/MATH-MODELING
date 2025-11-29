import pandas as pd
import numpy as np
import math
np.set_printoptions(precision=5,suppress=True)
Excelfile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")
var={
    "Y":"年份",
    "T1":"公路通车里程",
    "T2":"码头泊位个数",
    "T3":"客船",
    "T4":"客位",
    "T5":"铁路通车里程",
    "T6":"民航主要航线"

}
df=pd.read_excel(Excelfile,skiprows=0,sheet_name="Sheet3",usecols="A:G")
df.columns=["Y","T1","T2","T3","T4","T5","T6"]
Y=df["Y"].to_list()
T1=df["T1"].to_list()
T2=df["T2"].to_list()
T3=df["T3"].to_list()
T4=df["T4"].to_list()
T5=df["T5"].to_list()
T6=df["T6"].to_list()

df=pd.DataFrame({"T1":T1,"T2":T2,"T3":T3,"T4":T4,"T5":T5,"T6":T6})
cost_columns=[]
benefit_columns=["T1","T2","T3","T4","T5","T6"]
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
         return np.std(X_standard,axis=0)
     


     def relate(X_standard): #计算变量相关性与冲突性
          m,n=X_standard.shape
          copy=pd.DataFrame(X_standard,columns=df.columns[0:])
          cometrix=copy.corr()
          f=[0.0]*n
          co_values=cometrix.values
          for j in range(n):
               for i in range(n):
                    f[j]=(1-((co_values[i,j])**2)**0.5)+f[j]
               f[j]=float(f[j])
          return f

def critic(df,cost_columns,benefit_columns):
     X_standard=Critic.preproc(df,cost_columns,benefit_columns) #得到归一化矩阵
     n=X_standard.shape[1]
     R=Critic.conflict(X_standard) 
     f=Critic.relate(X_standard)   #相关性
     information_content=[R[i]*f[i] for i in range(n)]  #计算信息量
     weight=np.array([information_content[i]/sum(information_content) for i in range(6)]) #计算权重
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

     #得到相对贴近度（每年的交通指数）
     T=[float(dist_worst[i])/float((dist_best[i]+dist_worst[i])) for i in range(lenth)]
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


               


                          
        


