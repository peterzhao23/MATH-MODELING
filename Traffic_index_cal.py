import pandas as pd
import numpy as np
from critic import Critic,critic,topsis,entropy

np.set_printoptions(precision=4,suppress=True)

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
weight_critic=critic(df,cost_columns,benefit_columns)
T1=topsis(Critic.preproc(df,cost_columns,benefit_columns),weight_critic,11) 

#熵权法
weight_entro=entropy(df,cost_columns,benefit_columns)
T2=topsis(Critic.preproc(df,cost_columns,benefit_columns),weight_entro,11)
print(T1)