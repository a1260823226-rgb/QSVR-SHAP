import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===================== 核心样式调整 =====================
# 1. 设置字体为中宋西罗马（SimSun 对应中宋，兼容西文罗马体）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中宋（包含西罗马字符）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.family'] = 'serif'         # 西文强制使用衬线体（罗马体）

# 2. 构建数据并筛选（abs_r ≥ 0.1）
data = [
    ["(B_x/B_r_ion)/(X_x/X_r_ion)", -0.840675628, 2.01e-145, 0.840675628, True],
    ["(B_x / X_x)", -0.819905892, 2.02e-132, 0.819905892, True],
    ["(B_x - X_x)", -0.812222243, 4.96e-128, 0.812222243, True],
    ["(B_x/B_r_ion)", -0.800923778, 6.26e-122, 0.800923778, True],
    ["ratio_X_bx_bond_mean", -0.79675131, 8.94e-120, 0.79675131, True],
    ["(B_x / X_x)/(B_r_ion + X_r_ion)", -0.772366561, 4.03e-108, 0.772366561, True],
    ["ratio_ea_bx_bond_mean", -0.762593792, 7.51e-104, 0.762593792, True],
    ["B_x", -0.746938285, 2.01e-97, 0.746938285, True],
    ["(B_x/B_r_ion)-(X_x/X_r_ion)", -0.726299125, 1.22e-89, 0.726299125, True],
    ["delta_X_bx_bond_mean", 0.709932625, 5.86e-84, 0.709932625, True],
    ["(B_x - X_x)/(B_r_ion + X_r_ion)", -0.672181136, 3.00e-72, 0.672181136, True],
    ["(B_r_ion / X_r_ion)", 0.653443782, 4.63e-67, 0.653443782, True],
    ["B_r_ion", 0.639948518, 1.52e-63, 0.639948518, True],
    ["B_r_atom", 0.638630281, 3.27e-63, 0.638630281, True],
    ["(B_x/B_r_ion)*(X_x/X_r_ion)", -0.580229392, 6.56e-50, 0.580229392, True],
    ["B_ie", -0.54150872, 1.80e-42, 0.54150872, True],
    ["LUMO", 0.498224507, 3.19e-35, 0.498224507, True],
    ["(B_ea/B_r_ion)-(X_ea/X_r_ion)", -0.4951698, 9.48e-35, 0.4951698, True],
    ["(A_r_ion + X_r_ion)/(1.414*(B_r_ion + X_r_ion))", -0.487769376, 1.27e-33, 0.487769376, True],
    ["gap_AO", 0.44635819, 8.44e-28, 0.44635819, True],
    ["log(ratio_ea_bx_bond_var)", -0.428569549, 1.56e-25, 0.428569549, True],
    ["ratio_ea_bx_bond_var", -0.428569549, 1.56e-25, 0.428569549, True],
    ["delta_ea_bx_bond_mean", 0.355686398, 1.51e-17, 0.355686398, True],
    ["(B_ea - X_ea)", -0.348803856, 6.81e-17, 0.348803856, True],
    ["(B_r_ion + X_r_ion)", 0.336974004, 8.35e-16, 0.336974004, True],
    ["(B_ea/B_r_ion)*(X_ea/X_r_ion)", -0.330195836, 3.35e-15, 0.330195836, True],
    ["log(ratio_X_bx_bond_var)", -0.320756681, 2.19e-14, 0.320756681, True],
    ["ratio_X_bx_bond_var", -0.320756681, 2.19e-14, 0.320756681, True],
    ["(B_ea / X_ea)/(B_r_ion + X_r_ion)", -0.311652827, 1.26e-13, 0.311652827, True],
    ["(X_ea/X_r_ion)", 0.297783726, 1.61e-12, 0.297783726, False],
    ["X_r_ion", -0.296191289, 2.14e-12, 0.296191289, False],
    ["B_ea", -0.289842071, 6.53e-12, 0.289842071, False],
    ["X_M", -0.289592575, 6.81e-12, 0.289592575, False],
    ["B_M", -0.289229385, 7.26e-12, 0.289229385, False],
    ["B_N", -0.289229385, 7.26e-12, 0.289229385, False],
    ["X_N", -0.288218835, 8.64e-12, 0.288218835, False],
    ["ratio_ea_bx_bond_cv", -0.286765167, 1.11e-11, 0.286765167, False],
    ["log(ratio_ea_bx_bond_cv)", -0.286765167, 1.11e-11, 0.286765167, False],
    ["(B_ea / X_ea)", -0.285608078, 1.35e-11, 0.285608078, False],
    ["HOMO", -0.281057482, 2.92e-11, 0.281057482, False],
    ["(X_x/X_r_ion)", 0.280548479, 3.18e-11, 0.280548479, False],
    ["(B_ea/B_r_ion)", -0.279278991, 3.93e-11, 0.279278991, False],
    ["X_ea", 0.265331627, 3.75e-10, 0.265331627, False],
    ["(B_ea/B_r_ion)/(X_ea/X_r_ion)", -0.263277973, 5.17e-10, 0.263277973, False],
    ["(B_x/B_r_ion)+(X_x/X_r_ion)", -0.261236092, 7.10e-10, 0.261236092, False],
    ["log(ratio_X_bx_bond_cv)", -0.239462025, 1.76e-08, 0.239462025, False],
    ["ratio_X_bx_bond_cv", -0.239462025, 1.76e-08, 0.239462025, False],
    ["X_r_atom", -0.23879089, 1.94e-08, 0.23879089, False],
    ["(B_ea - X_ea)/(B_r_ion + X_r_ion)", -0.237895233, 2.19e-08, 0.237895233, False],
    ["X_ie", 0.224872991, 1.28e-07, 0.224872991, False],
    ["X_x", 0.224872991, 1.28e-07, 0.224872991, False],
    ["log(delta_X_bx_bond_cv)", -0.14212011, 0.000927161, 0.14212011, False],
    ["delta_X_bx_bond_cv", -0.14212011, 0.000927161, 0.14212011, False]
]
df = pd.DataFrame(data, columns=['feature', 'spearman_r', 'p_value', 'abs_r', 'pass_threshold'])

# 筛选abs_r ≥ 0.1的特征
df_filtered = df[df['abs_r'] >= 0.1].sort_values('abs_r', ascending=False)
features = df_filtered['feature'].values
spearman_r = df_filtered['spearman_r'].values

# 构造特征间的相关矩阵
corr_matrix = np.outer(spearman_r, spearman_r)

# ===================== 核心样式调整 =====================
# 3. 创建正方形画布 + 保证颜色条与热力图等高
plt.figure(figsize=(20, 20))  # 正方形画布

# 绘制热力图（关键：通过cbar_kws设置颜色条位置和高度）
ax = sns.heatmap(
    corr_matrix,
    annot=True,                  # 显示数值
    fmt='.3f',                   # 数值格式
    cmap='RdBu_r',               # 颜色映射
    cbar=True,                   # 显示颜色条
    yticklabels=features,        # y轴标签
    xticklabels=features,        # x轴标签
    vmin=-1, vmax=1,             # 颜色范围-1到1
    linewidths=0.5,              # 网格线宽度
    annot_kws={'size': 6},       # 标注字体大小
    # 2. 关键：让颜色条与热力图完全等高（通过location和pad控制位置，shrink=1保证高度）
    cbar_kws={
        'location': 'right',     # 颜色条在右侧
        'pad': 0.02,             # 颜色条与热力图的间距
        'shrink': 1,             # 颜色条高度=热力图高度（核心参数）
        'aspect': 40,            # 颜色条宽高比（适配整体尺寸）
        'ticks': [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]  # 颜色条刻度
    }
)

# 设置标题和标签（中宋西罗马字体生效）
plt.title('Spearman Correlation Matrix (abs_r ≥ 0.1)', fontsize=18, pad=25)
plt.ylabel('Features', fontsize=14)
plt.xlabel('Features', fontsize=14)

# 调整标签样式
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)   # y轴标签不旋转
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=45, ha='right')  # x轴标签旋转45度

# 强制调整布局，避免标签截断
plt.tight_layout()

# 保存高清图片（正方形）
plt.savefig('spearman_correlation_final.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()