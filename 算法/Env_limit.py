import pandas as pd
import numpy as np
from critic import entropy,critic,topsis,Critic,normalize
from multiple_variants_linear import tourismmodel
ExcelFile=pd.ExcelFile("三亚过夜游客统计新表.xlsx")
df=pd.read_excel(ExcelFile,skiprows=0,usecols="B:S",sheet_name="Sheet8")
df.columns=["人均生产总值（元）","常住人口 (万人)","城镇化率 (%)","人口密度（人/平方公里）",
            "森林覆盖率","城镇生活污水集中处理率","PM2.5 (μg/m³)"	,"NO₂ (μg/m³)","SO₂ (μg/m³)","地表水达标率",
            "水资源总量（亿立方米）","每万元GDP水耗（立方米）","人均用水量（平方米/人）","人均耕地面积（公顷）",
            "游客数量（万人）","交通指数","环境指数","海南省A级景区数量"]
cost_cols=df.columns.to_list()
benefit_cols=[]
EL_=pd.DataFrame(critic(df,cost_cols,benefit_cols).reshape(1,-1),columns=df.columns.to_list())
EL1_df=EL_[["人均生产总值（元）","常住人口 (万人)","城镇化率 (%)","人口密度（人/平方公里）"]]
EL2_df=EL_[["森林覆盖率","城镇生活污水集中处理率","PM2.5 (μg/m³)"	,"NO₂ (μg/m³)","SO₂ (μg/m³)","地表水达标率"]]
EL3_df=EL_[["水资源总量（亿立方米）","每万元GDP水耗（立方米）","人均用水量（平方米/人）","人均耕地面积（公顷）"]]
EL4_df=EL_[["游客数量（万人）","交通指数","环境指数","海南省A级景区数量"]]
EL1_=np.sum(EL1_df.values,axis=1) #经济社会承载力权重
EL2_=np.sum(EL2_df.values,axis=1)    #环境承载力权重
EL3_=np.sum(EL3_df.values,axis=1) #资源承载力
EL4_=np.sum(EL4_df.values,axis=1)#旅游承载力
weight_EL1_=EL1_df.values/EL1_
weight_EL2_=EL2_df.values/EL2_
weight_EL3_=EL3_df.values/EL3_
weight_EL4_=EL4_df.values/EL4_

EL1_top=topsis(Critic.preproc(df[["人均生产总值（元）","常住人口 (万人)","城镇化率 (%)","人口密度（人/平方公里）"]],cost_columns=EL1_df.columns.to_list(),benefit_columns=[]),weight_EL1_,11)
EL2_top=topsis(Critic.preproc(df[["森林覆盖率","城镇生活污水集中处理率","PM2.5 (μg/m³)"	,"NO₂ (μg/m³)","SO₂ (μg/m³)","地表水达标率"]],cost_columns=EL2_df.columns.to_list(),benefit_columns=[]),weight_EL2_,11)
EL3_top=topsis(Critic.preproc(df[["水资源总量（亿立方米）","每万元GDP水耗（立方米）","人均用水量（平方米/人）","人均耕地面积（公顷）"]],cost_columns=EL3_df.columns.to_list(),benefit_columns=[]),weight_EL3_,11)
EL4_top=topsis(Critic.preproc(df[["游客数量（万人）","交通指数","环境指数","海南省A级景区数量"]],cost_columns=EL4_df.columns.to_list(),benefit_columns=[]),weight_EL4_,11)
#主因素突出算法,#归一化处理
EL_all=normalize(EL1_top*EL2_top*EL3_top*EL4_top)
print(EL_.values)
