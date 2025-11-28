import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
Excelfile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")
var={
    "Y":"年份",
    "T1":"公路通车里程",
    "T2":"码头泊位个数",
    "T3":"客船",
    "T4":"铁路通车里程",
    "T5":"民航主要航线"

}
df=pd.read_excel(Excelfile,skiprows=0,sheet_name="Sheet3",usecols="A:G")
df.columns=["Y","T1","T2","T3","T4","T5"]
Y=df["Y"].to_list()
T1=df["T1"].to_list()
T2=df["T2"].to_list()
T3=df["T3"].to_list()
T4=df["T4"].to_list()
T5=df["T5"].to_list()

df=pd.DataFrame({"Y":Y,"T1":T1,"T2":T2,"T3":T3,"T4":T4,"T5":T5})
#创建cttic类
class critic_anl:
    def __init__(self):
        self.scaler=MinMaxScaler()


    def preproc(self,df):
        X=
