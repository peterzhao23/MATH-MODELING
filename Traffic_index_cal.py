import pandas as pd
import numpy as np
from critic import Critic,critic,topsis

np.set_printoptions(precision=4,suppress=True)

Excelfile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")

df=pd.read_excel(Excelfile,skiprows=0,sheet_name="Sheet3",usecols="B:F")
df.columns=["公路通车里程","客船","客位","铁路通车里程","民航主要航线"]

cost_columns=[]
benefit_columns=df.columns.to_list()
weight_critic=critic(df,cost_columns,benefit_columns)
T1=topsis(Critic.preproc(df,cost_columns,benefit_columns),weight_critic,11) 

print(T1)