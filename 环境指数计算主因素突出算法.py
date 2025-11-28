import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler #数据处理模块
np.set_printoptions(precision=3,suppress=True)
Excelfile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")

df=pd.read_excel(Excelfile,skiprows=0,sheet_name="Sheet5",usecols="B:G")
X=pd.DataFrame(df).values

standard=MinMaxScaler()
X_scaled=standard.fit_transform(X)
print(X_scaled)


