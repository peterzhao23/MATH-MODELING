import pandas as pd
import numpy as np
np.set_printoptions(precision=3,suppress=True)
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
cost_columns=["T1","T3","T5"]
benefit_columns=["T2","T4","T6"]
#创建预处理函数
def preproc(df, cost_columns):
     X=df.values.astype(float)#将X转化为np数组，转化为浮点数，若为整数，最后结果只有0，1
     xmin=X.min(axis=0)
     xmax=X.max(axis=0)
     xmaxmin=xmax-xmin
     m,n=X.shape
     for i in range(m):
          for j in range(n):
               if df.columns[j] in cost_columns:
                    X[i,j] = float((xmax[j] - X[i,j]) / xmaxmin[j])
               else:
                    X[i,j] =  float((X[i,j] - xmin[j]) / xmaxmin[j])
     return X



def conflict(X_standard):  #计算标准差（对比强度）
     X=X_standard.copy()
     m,n=X_standard.shape
     avr=[0.0]*n
     for j in range(n):
          for i in range(m):
            avr[j]=X[i,j]+avr[j]
     avr=[float(x)/m for x in avr]
     avr1=[0.0]*n
     for j in range(n):
          for i in range(m):
               avr1[j]=(X[i,j]-avr[j])**2+avr1[j]
          avr1[j]=float((avr1[j]/(m-1))**0.5)
     return avr1 


def relate(X_standard): #计算变量相关性
     m,n=X_standard.shape
     copy=pd.DataFrame(X_standard,columns=df.columns[0:])
     cometrix=copy.corr()
     f=[0.0]*n
     co_values=cometrix.values
     for j in range(n):
          for i in range(n):
               f[j]=(1-((co_values[i,j])**2)**0.5)+f[j]
          f[j]=float(f[j])-1
     return f


X_standard=preproc(df,cost_columns) #得到归一化矩阵
R=conflict(X_standard) #冲突性
f=relate(X_standard)   #相关性
information_content=[R[i]*f[i] for i in range(6)]  #计算信息量
weight_row=[information_content[i]/sum(information_content) for i in range(6)] #计算权重

#构造加权规范矩阵
weight_matrix=np.array(weight_row).reshape(1,-1) #将权重转换成矩阵，reshape保留原列表行与列（1表示行，-1表示列）
X_weight=X_standard*weight_matrix
           
#确定正理想解与负理想解
Vmin=X_weight.min(axis=0)
Vmax=X_weight.max(axis=0)
print(weight_matrix)                          
        
#计算每个评价对象到理想解的距离
dist_best = np.sqrt(np.sum((X_weight - Vmax) ** 2, axis=1))
dist_worst = np.sqrt(np.sum((X_weight - Vmin) ** 2, axis=1))

#得到相对贴近度（每年的交通指数）
T=[float(dist_worst[i])/float((dist_best[i]+dist_worst[i])) for i in range(11)]
print(T)
