import pandas as pd
import numpy as np
from critic import entropy
from multiple_variants_linear import tourismmodel
ExcelFile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")
df=pd.read_excel(ExcelFile,skiprows=0,usecols="B:O",sheet_name="Sheet8")
df.columns=["人均生产总值（元）","常住人口 (万人)","城镇化率 (%)","人口密度（人/平方公里）","森林覆盖率","城镇生活污水集中处理率","PM2.5 (μg/m³)"	,"NO₂ (μg/m³)","SO₂ (μg/m³)","地表水达标率","水资源总量（亿立方米）","每万元GDP水耗（立方米）","人均用水量（平方米/人）","人均耕地面积（公顷）"]

tm=tourismmodel()
tm.exporatory_data_analysis(df)








cost_cols=[i for i in range(len(df.columns))]
benefit_cols=[]
EL=pd.DataFrame(entropy(df,cost_cols,benefit_cols).reshape(1,-1),columns=df.columns.to_list())
EL1=np.sum(entropy(df,cost_cols,benefit_cols)[0:4]) #经济社会承载力权重
EL2=np.sum(entropy(df,cost_cols,benefit_cols)[4:10])    #环境承载力权重
EL3=np.sum(entropy(df,cost_cols,benefit_cols)[1:2])
                                              