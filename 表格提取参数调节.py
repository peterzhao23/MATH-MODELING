
df = pd.read_excel(
    'your_file.xlsx',
    sheet_name='Sheet1',      # 工作表名称或索引
    header=0,                 # 使用第几行作为列名（0表示第一行）
    names=['col1', 'col2', 'col3'],  # 自定义列名
    index_col=0,              # 使用第几列作为索引
    usecols='A:C',            # 只读取特定列（A到C列）
    nrows=100,                # 只读取前100行
    skiprows=[0, 2, 3],       # 跳过指定行
    dtype={'col1': str, 'col2': float}  # 指定列的数据类型
)

读取工作表



X1=df[cost_columns]   #成本型指标
        X2=df[benefit_columns]   #效益型指标
        m1,n1=X1.shape
        m2,n2=X2.shape
        x1min=X1.min(axis=0)
        x1max=X1.max(axis=0)
        x1maxmin=x1max-x1min
        x2min=X2.min(axis=0)
        x2max=X2.max(axis=0)
        x2maxmin=x2max-x2min
        
        for i in range(m1):
            for j in range(n1):
                X1[i, j] = (x1max[j] - X1[i, j]) / x1maxmin[j] #逆向处理
        for i in range(m2):
            for j in range(n2):
                X2[i,j] = (X2[i,j]-x2min[j])/ x2maxmin[j] #正向处理
