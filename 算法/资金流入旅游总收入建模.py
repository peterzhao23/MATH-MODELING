import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
years = list(range(2013, 2023))
tourism_revenue = [217.2, 258, 291.9, 305.5, 370.4, 469.91, 525.33, 417.73, 743.2, 431.54]
gov_expenditure = [1852589, 1974596, 2039427, 2428284, 2209223, 2585252, 2686338, 2635273, 2886399, 3233228]

P_current = np.array(tourism_revenue[:-1])
P_next_actual = np.array(tourism_revenue[1:])
AR_next = np.array(gov_expenditure[1:])

print("原始数据R²基准:")
baseline_r2 = r2_score(P_next_actual, P_current)
print(f"用本年预测下年 R² = {baseline_r2:.4f}")

# 定义函数形式
def linear_func(X, a, b):
    P_curr, AR_nxt = X
    return a * P_curr + b * (AR_nxt/1000000)

def power_func(X, a, b, c):
    P_curr, AR_nxt = X
    return a * (P_curr ** b) * ((AR_nxt/1000000) ** c)

def exp_func(X, a, b, c):
    P_curr, AR_nxt = X
    return a * np.exp(b * P_curr + c * (AR_nxt/1000000))

def log_func(X, a, b, c):
    P_curr, AR_nxt = X
    return a * np.log1p(P_curr) + b * np.log1p(AR_nxt/1000000) + c

# 分段拟合
print("\n" + "="*60)
print("成功拟合的函数及其参数")
print("="*60)

successful_models = []

# 疫情前模型
pre_covid_mask = np.array([year < 2020 for year in years[1:]])
if sum(pre_covid_mask) >= 3:
    P_curr_pre = P_current[pre_covid_mask]
    P_next_pre = P_next_actual[pre_covid_mask]
    AR_next_pre = AR_next[pre_covid_mask]
    
    print("\n疫情前模型 (2014-2019):")
    print("-" * 40)
    
    # 线性函数
    try:
        popt, pcov = curve_fit(linear_func, (P_curr_pre, AR_next_pre), P_next_pre, p0=[1, 1], maxfev=5000)
        pred = linear_func((P_curr_pre, AR_next_pre), *popt)
        r2 = r2_score(P_next_pre, pred)
        
        # 输出带参数值的函数形式
        print(f"线性函数:")
        print(f"  P_next = {popt[0]:.6f} * P_curr + {popt[1]:.6f} * (AR_next/1000000)")
        print(f"  R² = {r2:.4f}")
        
        successful_models.append(('疫情前线性模型', linear_func, popt, r2))
    except Exception as e:
        print(f"线性函数拟合失败: {e}")
    
    # 幂函数
    try:
        popt, pcov = curve_fit(power_func, (P_curr_pre, AR_next_pre), P_next_pre, p0=[1, 0.5, 0.5], maxfev=5000)
        pred = power_func((P_curr_pre, AR_next_pre), *popt)
        r2 = r2_score(P_next_pre, pred)
        
        # 输出带参数值的函数形式
        print(f"\n幂函数:")
        print(f"  P_next = {popt[0]:.6f} * (P_curr^{popt[1]:.6f}) * ((AR_next/1000000)^{popt[2]:.6f})")
        print(f"  R² = {r2:.4f}")
        
        successful_models.append(('疫情前幂函数模型', power_func, popt, r2))
    except Exception as e:
        print(f"幂函数拟合失败: {e}")
    
    # 指数函数
    try:
        popt, pcov = curve_fit(exp_func, (P_curr_pre, AR_next_pre), P_next_pre, p0=[100, 0.01, 0.01], maxfev=5000)
        pred = exp_func((P_curr_pre, AR_next_pre), *popt)
        r2 = r2_score(P_next_pre, pred)
        
        # 输出带参数值的函数形式
        print(f"\n指数函数:")
        print(f"  P_next = {popt[0]:.6f} * exp({popt[1]:.6f} * P_curr + {popt[2]:.6f} * (AR_next/1000000))")
        print(f"  R² = {r2:.4f}")
        
        successful_models.append(('疫情前指数模型', exp_func, popt, r2))
    except Exception as e:
        print(f"指数函数拟合失败: {e}")

# 疫情后模型
post_covid_mask = ~pre_covid_mask
if sum(post_covid_mask) >= 2:
    P_curr_post = P_current[post_covid_mask]
    P_next_post = P_next_actual[post_covid_mask]
    AR_next_post = AR_next[post_covid_mask]
    
    print("\n疫情后模型 (2020-2022):")
    print("-" * 40)
    
    # 线性函数
    try:
        popt, pcov = curve_fit(linear_func, (P_curr_post, AR_next_post), P_next_post, p0=[1, 1], maxfev=5000)
        pred = linear_func((P_curr_post, AR_next_post), *popt)
        r2 = r2_score(P_next_post, pred)
        
        # 输出带参数值的函数形式
        print(f"线性函数:")
        print(f"  P_next = {popt[0]:.6f} * P_curr + {popt[1]:.6f} * (AR_next/1000000)")
        print(f"  R² = {r2:.4f}")
        
        successful_models.append(('疫情后线性模型', linear_func, popt, r2))
    except Exception as e:
        print(f"线性函数拟合失败: {e}")
    
    # 对数函数
    try:
        popt, pcov = curve_fit(log_func, (P_curr_post, AR_next_post), P_next_post, p0=[100, 10, 100], maxfev=5000)
        pred = log_func((P_curr_post, AR_next_post), *popt)
        r2 = r2_score(P_next_post, pred)
        
        # 输出带参数值的函数形式
        print(f"\n对数函数:")
        print(f"  P_next = {popt[0]:.6f} * ln(1+P_curr) + {popt[1]:.6f} * ln(1+AR_next/1000000) + {popt[2]:.6f}")
        print(f"  R² = {r2:.4f}")
        
        successful_models.append(('疫情后对数模型', log_func, popt, r2))
    except Exception as e:
        print(f"对数函数拟合失败: {e}")

# 机器学习模型
print("\n机器学习模型:")
print("-" * 40)

# 准备特征矩阵
X = np.column_stack([
    P_current,
    AR_next / 1000000,
    np.arange(len(P_current))
])
y = P_next_actual

# 线性回归
lr = LinearRegression()
lr.fit(X, y)
lr_r2 = r2_score(y, lr.predict(X))

# 输出带参数值的函数形式
print(f"线性回归:")
print(f"  P_next = {lr.intercept_:.6f} + {lr.coef_[0]:.6f} * P_curr + {lr.coef_[1]:.6f} * (AR_next/1000000) + {lr.coef_[2]:.6f} * t")
print(f"  R² = {lr_r2:.4f}")

successful_models.append(('线性回归', '机器学习', lr.coef_, lr_r2))

# 随机森林
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_r2 = r2_score(y, rf.predict(X))

# 输出函数形式
print(f"\n随机森林:")
print(f"  R² = {rf_r2:.4f}")
print(f"  特征重要性: P_curr = {rf.feature_importances_[0]:.4f}, AR_next = {rf.feature_importances_[1]:.4f}, 时间 = {rf.feature_importances_[2]:.4f}")

successful_models.append(('随机森林', '机器学习', rf.feature_importances_, rf_r2))

# 总结所有成功模型
print("\n" + "="*60)
print("所有成功拟合的模型总结")
print("="*60)

print(f"\n总共成功拟合 {len(successful_models)} 个模型:")
for i, (name, func_type, params, r2) in enumerate(successful_models, 1):
    print(f"\n{i}. {name}")
    print(f"   R²: {r2:.4f}")
    
    if func_type == '机器学习':
        if name == '线性回归':
            print(f"   具体函数: P_next = {lr.intercept_:.6f} + {params[0]:.6f} * P_curr + {params[1]:.6f} * AR_next/{1000000} + {params[2]:.6f} * t")
        else:
            print(f"   特征重要性: P_curr = {params[0]:.4f}, AR_next = {params[1]:.4f}, 时间 = {params[2]:.4f}")
    else:
        # 根据函数类型输出不同的带参数值的函数形式
        if name == '疫情前线性模型' or name == '疫情后线性模型':
            print(f"   具体函数: P_next = {params[0]:.6f} * P_curr + {params[1]:.6f} * AR_next/{1000000}")
        elif name == '疫情前幂函数模型':
            print(f"   具体函数: P_next = {params[0]:.6f} * P_curr^{params[1]:.6f} * (AR_next/{1000000})^{params[2]:.6f}")
        elif name == '疫情前指数模型':
            print(f"   具体函数: P_next = {params[0]:.6f} * exp({params[1]:.6f} * P_curr + {params[2]:.6f} * AR_next/{1000000})")
        elif name == '疫情后对数模型':
            print(f"   具体函数: P_next = {params[0]:.6f} * ln(1+P_curr) + {params[1]:.6f} * ln(1+AR_next/{1000000}) + {params[2]:.6f}")

# 绘制包含全部拟合函数及相应优度的图
print(f"\n绘制包含全部拟合函数及相应优度的图...")

# 创建图像
plt.figure(figsize=(15, 10))

# 绘制实际值
years_plot = years[1:]  # 使用2014-2022年
plt.plot(years_plot, P_next_actual, 'ko-', linewidth=3, markersize=10, label='实际旅游收入', zorder=10)

# 定义颜色和线型
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
line_styles = ['-', '--', '-.', ':', '-', '--']

# 绘制每个模型的拟合曲线
for i, (name, func_type, params, r2) in enumerate(successful_models):
    color = colors[i % len(colors)]
    line_style = line_styles[i % len(line_styles)]
    
    # 计算预测值
    if func_type == '机器学习':
        if name == '线性回归':
            predictions = lr.predict(X)
        else:
            predictions = rf.predict(X)
    else:
        predictions = func_type((P_current, AR_next), *params)
    
    # 绘制拟合曲线
    plt.plot(years_plot, predictions, color=color, linestyle=line_style, 
             linewidth=2, label=f'{name} (R²={r2:.4f})')

# 设置图表属性
plt.xlabel('年份', fontsize=14)
plt.ylabel('旅游收入 (亿元)', fontsize=14)
plt.title('全部拟合函数对比及优度', fontsize=16)
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)

# 添加数据标签
for i, (year, actual) in enumerate(zip(years_plot, P_next_actual)):
    plt.annotate(f'{actual:.1f}', (year, actual), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()

# 使用最佳模型进行预测的函数
def predict_with_model(model_info, P_curr, AR_nxt):
    name, func_type, params, r2 = model_info
    
    if func_type == '机器学习':
        if name == '线性回归':
            # params 是系数数组
            return params[0] * P_curr + params[1] * (AR_nxt/1000000) + params[2] * (len(P_current)) + lr.intercept_
        else:
            # 随机森林预测需要特征矩阵
            X_pred = np.array([[P_curr, AR_nxt/1000000, len(P_current)]])
            return rf.predict(X_pred)[0]
    else:
        # 函数模型
        return func_type((P_curr, AR_nxt), *params)

# 示例：使用最佳模型预测2023年
if successful_models:
    best_model = max(successful_models, key=lambda x: x[3])
    print(f"\n最佳模型: {best_model[0]} (R² = {best_model[3]:.4f})")
    
    # 使用最佳模型预测2023年
    P_2023_best = predict_with_model(best_model, tourism_revenue[-1], gov_expenditure[-1])
    print(f"使用最佳模型预测2023年旅游收入: {P_2023_best:.2f} 亿元")

print(f"\n建模完成！共成功拟合 {len(successful_models)} 个模型。")