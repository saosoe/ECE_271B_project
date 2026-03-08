import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 读取数据并清理缺失值
df = pd.read_excel('/Users/qibuguojiuhuolala/Desktop/Project/Dataset/Unnormalized_data.xlsx')
df = df.dropna()

# 2. 划定范围：候选自变量 (A列到K列，即前11列) 与 目标因变量
candidate_cols = df.columns[0:11].tolist() 
target_col = 'diastolic pressure'  # 如果你想预测舒张压，请改为 'diastolic pressure'
y = df[target_col]

# 设置共线性严格阈值
VIF_THRESHOLD = 5.0  

best_r2 = -1
best_combo = None
best_vif_details = {}

# 3. 遍历所有 3 个变量的组合 (总计 165 种)
for combo in itertools.combinations(candidate_cols, 3):
    X_subset = df[list(combo)]
    X_with_const = sm.add_constant(X_subset)
    
    # 计算当前组合的 VIF
    # range(1, 4) 是为了跳过第 0 列的常数项 (const)
    vifs = [variance_inflation_factor(X_with_const.values, i) for i in range(1, 4)]
    
    # 4. 如果该组合中所有变量的 VIF 均小于阈值，则评估其预测“质量”
    if max(vifs) < VIF_THRESHOLD:
        # 建立 OLS 线性回归模型，评估预测能力
        model = sm.OLS(y, X_with_const).fit()
        
        # 如果当前的 R-squared 更高，则记录为最佳组合
        if model.rsquared > best_r2:
            best_r2 = model.rsquared
            best_combo = combo
            # 将变量名和对应的 VIF 打包成字典
            best_vif_details = dict(zip(combo, vifs))

# 5. 输出最终结果
if best_combo:
    print(f"🎯 选出的最佳 3 个变量是: {best_combo}")
    print(f"📈 它们对 {target_col} 的联合解释度 (R-squared) 为: {best_r2:.4f}")
    print("-" * 30)
    print("各变量的 VIF 详情 (共线性越低越好):")
    for var, vif_val in best_vif_details.items():
        print(f" - {var}: {vif_val:.2f}")
else:
    print(f"⚠️ 在当前的阈值 (VIF < {VIF_THRESHOLD}) 下，没有找到符合要求的 3 变量组合。")
    print("建议：尝试将 VIF_THRESHOLD 稍微调高（例如调到 10），或者检查数据是否存在极端共线性。")