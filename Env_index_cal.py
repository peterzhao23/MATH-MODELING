import pandas as pd
import numpy as np
import math



from critic import Critic,critic,topsis,entropy,standardlize,normalize
np.set_printoptions(precision=4,suppress=True)



Excelfile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")

df=pd.read_excel(Excelfile,skiprows=0,sheet_name="Sheet5",usecols="B:G")
df.columns=["绿化覆盖率", "绿地率", "人均公共绿地面积", "森林覆盖率", "地表水达标率", "污水处理率"]
X=df.values



Xstandard=standardlize(X)
#采用主因素突出算法
E=np.prod(Xstandard,axis=1)
#将数据归一化
E1=normalize(E)
#过于突出极端值的影响使数据过于离散，舍去

cost_columns=[]
benefit_columns=df.columns.to_list()
#熵权法
weight_entro=entropy(df,cost_columns,benefit_columns)
E2=topsis(Critic.preproc(df,cost_columns,benefit_columns),weight_entro,11)

print(E2)

