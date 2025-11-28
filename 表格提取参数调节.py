
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
