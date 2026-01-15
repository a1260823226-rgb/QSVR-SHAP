# ===================== 1. 导入所需库 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 相关性分析库
from scipy.stats import spearmanr, kendalltau
import dcor
import pingouin as pg
from scipy.stats import pearsonr

# 特征选择库
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# SHAP分析库
import shap
import xgboost as xgb

# ===================== 2. 配置路径和全局参数 =====================
# 数据路径
DATA_PATH = r'C:\Users\Administrator\Desktop\newmat\cleaned_mix-halide-perovskite-ml-bandgap.csv'
# 结果保存路径
OUTPUT_DIR = r'C:\Users\Administrator\Desktop\newmat\feature_analysis_results'
# 目标变量
TARGET_COL = 'bandgap'
# 忽略的非数值列
IGNORE_COLS = ['structure', 'A_element', 'B_element', 'X_elements']
# 筛选阈值（可根据需求调整）
THRESHOLDS = {
    'spearman': 0.3,
    'kendall': 0.2,
    'mutual_info': 0.1,
    'distance_corr': 0.4,
    'vif': 5
}

# 创建结果保存目录
import os
import re

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置中文字体（解决图表中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 辅助函数：清理特征名称（解决文件名非法字符问题） =====================
def clean_feature_name(feature_name):
    """清理特征名称中的特殊字符，使其可作为合法文件名"""
    # 替换Windows不允许的字符：/ \ : * ? " < > | 空格
    illegal_chars = r'[\/:*?"<>|()\[\]\+\-\*\/ ]'  # 转义特殊正则字符
    clean_name = re.sub(illegal_chars, '_', feature_name)
    # 移除连续下划线，避免过长
    clean_name = re.sub('_+', '_', clean_name)
    # 移除首尾下划线
    clean_name = clean_name.strip('_')
    # 限制长度
    if len(clean_name) > 50:
        clean_name = clean_name[:50]
    return clean_name


# ===================== 3. 数据加载和预处理 =====================
def load_and_preprocess_data():
    """加载数据并预处理：分离数值特征和目标变量，处理缺失值"""
    print("=== 加载并预处理数据 ===")
    # 加载数据
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    # 查看数据基本信息
    print(f"数据集形状：{df.shape}")
    print(f"目标变量({TARGET_COL})范围：{df[TARGET_COL].min():.4f} - {df[TARGET_COL].max():.4f} eV")

    # 分离数值特征和目标变量
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in IGNORE_COLS + [TARGET_COL]]

    # 处理缺失值（本数据集无缺失，仅做兜底）
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df[TARGET_COL]

    # 标准化特征（用于后续模型）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)

    print(f"数值特征数量：{len(numeric_cols)}")
    print(f"特征列表前10个：{numeric_cols[:10]}")

    return df, X, X_scaled_df, y, numeric_cols


# ===================== 4. 方法1：Spearman秩相关系数 =====================
def spearman_analysis(X, y, feature_names):
    """Spearman秩相关系数分析"""
    print("\n=== 方法1：Spearman秩相关系数分析 ===")
    spearman_results = []

    for col in feature_names:
        corr, p_value = spearmanr(X[col], y)
        spearman_results.append({
            'feature': col,
            'spearman_r': corr,
            'p_value': p_value,
            'abs_r': abs(corr),
            'pass_threshold': abs(corr) >= THRESHOLDS['spearman']
        })

    # 转换为DataFrame并排序
    spearman_df = pd.DataFrame(spearman_results)
    spearman_df = spearman_df.sort_values('abs_r', ascending=False).reset_index(drop=True)

    # 保存结果
    spearman_df.to_csv(os.path.join(OUTPUT_DIR, 'spearman_results.csv'), index=False, encoding='utf-8')

    # 可视化前20个特征
    top20 = spearman_df.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='spearman_r', y='feature', data=top20, palette='coolwarm')
    plt.axvline(x=THRESHOLDS['spearman'], color='red', linestyle='--', label=f'阈值={THRESHOLDS["spearman"]}')
    plt.axvline(x=-THRESHOLDS['spearman'], color='red', linestyle='--')
    plt.title('Spearman秩相关系数（Top20特征）', fontsize=14)
    plt.xlabel('Spearman r')
    plt.ylabel('特征')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spearman_top20.png'), dpi=300)
    plt.close()

    # 输出统计信息
    pass_count = spearman_df['pass_threshold'].sum()
    print(f"Spearman系数>={THRESHOLDS['spearman']}的特征数量：{pass_count}/{len(feature_names)}")
    print(f"最高相关特征：{spearman_df.iloc[0]['feature']} (r={spearman_df.iloc[0]['spearman_r']:.4f})")

    return spearman_df


# ===================== 5. 方法2：Kendall Tau相关系数 =====================
def kendall_analysis(X, y, feature_names):
    """Kendall Tau相关系数分析"""
    print("\n=== 方法2：Kendall Tau相关系数分析 ===")
    kendall_results = []

    for col in feature_names:
        corr, p_value = kendalltau(X[col], y)
        kendall_results.append({
            'feature': col,
            'kendall_tau': corr,
            'p_value': p_value,
            'abs_tau': abs(corr),
            'pass_threshold': abs(corr) >= THRESHOLDS['kendall']
        })

    # 转换为DataFrame并排序
    kendall_df = pd.DataFrame(kendall_results)
    kendall_df = kendall_df.sort_values('abs_tau', ascending=False).reset_index(drop=True)

    # 保存结果
    kendall_df.to_csv(os.path.join(OUTPUT_DIR, 'kendall_results.csv'), index=False, encoding='utf-8')

    # 可视化前20个特征
    top20 = kendall_df.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='kendall_tau', y='feature', data=top20, palette='viridis')
    plt.axvline(x=THRESHOLDS['kendall'], color='red', linestyle='--', label=f'阈值={THRESHOLDS["kendall"]}')
    plt.axvline(x=-THRESHOLDS['kendall'], color='red', linestyle='--')
    plt.title('Kendall Tau相关系数（Top20特征）', fontsize=14)
    plt.xlabel('Kendall τ')
    plt.ylabel('特征')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kendall_top20.png'), dpi=300)
    plt.close()

    # 输出统计信息
    pass_count = kendall_df['pass_threshold'].sum()
    print(f"Kendall系数>={THRESHOLDS['kendall']}的特征数量：{pass_count}/{len(feature_names)}")
    print(f"最高相关特征：{kendall_df.iloc[0]['feature']} (τ={kendall_df.iloc[0]['kendall_tau']:.4f})")

    return kendall_df


# ===================== 6. 方法3：互信息（Mutual Information） =====================
def mutual_info_analysis(X, y, feature_names):
    """互信息分析"""
    print("\n=== 方法3：互信息分析 ===")
    # 计算互信息值
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # 整理结果
    mi_results = []
    for i, col in enumerate(feature_names):
        mi_results.append({
            'feature': col,
            'mutual_info': mi_scores[i],
            'pass_threshold': mi_scores[i] >= THRESHOLDS['mutual_info']
        })

    # 转换为DataFrame并排序
    mi_df = pd.DataFrame(mi_results)
    mi_df = mi_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)

    # 保存结果
    mi_df.to_csv(os.path.join(OUTPUT_DIR, 'mutual_info_results.csv'), index=False, encoding='utf-8')

    # 可视化前20个特征
    top20 = mi_df.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mutual_info', y='feature', data=top20, palette='plasma')
    plt.axvline(x=THRESHOLDS['mutual_info'], color='red', linestyle='--', label=f'阈值={THRESHOLDS["mutual_info"]}')
    plt.title('互信息值（Top20特征）', fontsize=14)
    plt.xlabel('互信息值')
    plt.ylabel('特征')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mutual_info_top20.png'), dpi=300)
    plt.close()

    # 输出统计信息
    pass_count = mi_df['pass_threshold'].sum()
    print(f"互信息值>={THRESHOLDS['mutual_info']}的特征数量：{pass_count}/{len(feature_names)}")
    print(f"最高互信息特征：{mi_df.iloc[0]['feature']} (MI={mi_df.iloc[0]['mutual_info']:.4f})")

    return mi_df


# ===================== 7. 方法4：距离相关系数（Distance Correlation） =====================
def distance_corr_analysis(X, y, feature_names):
    """距离相关系数分析"""
    print("\n=== 方法4：距离相关系数分析 ===")
    dc_results = []

    for col in feature_names:
        # 计算距离相关系数
        dc = dcor.distance_correlation(X[col], y)
        dc_results.append({
            'feature': col,
            'distance_corr': dc,
            'pass_threshold': dc >= THRESHOLDS['distance_corr']
        })

    # 转换为DataFrame并排序
    dc_df = pd.DataFrame(dc_results)
    dc_df = dc_df.sort_values('distance_corr', ascending=False).reset_index(drop=True)

    # 保存结果
    dc_df.to_csv(os.path.join(OUTPUT_DIR, 'distance_corr_results.csv'), index=False, encoding='utf-8')

    # 可视化前20个特征
    top20 = dc_df.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='distance_corr', y='feature', data=top20, palette='cividis')
    plt.axvline(x=THRESHOLDS['distance_corr'], color='red', linestyle='--', label=f'阈值={THRESHOLDS["distance_corr"]}')
    plt.title('距离相关系数（Top20特征）', fontsize=14)
    plt.xlabel('Distance Correlation')
    plt.ylabel('特征')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distance_corr_top20.png'), dpi=300)
    plt.close()

    # 输出统计信息
    pass_count = dc_df['pass_threshold'].sum()
    print(f"距离相关系数>={THRESHOLDS['distance_corr']}的特征数量：{pass_count}/{len(feature_names)}")
    print(f"最高距离相关特征：{dc_df.iloc[0]['feature']} (DC={dc_df.iloc[0]['distance_corr']:.4f})")

    return dc_df


# ===================== 8. 方法5：偏相关分析（Partial Correlation）- 优化计算逻辑 =====================
def partial_corr_analysis(df, feature_names, target_col):
    """偏相关分析（优化计算逻辑，扩大可选控制变量）"""
    print("\n=== 方法5：偏相关分析 ===")
    # 选择核心控制变量（优先选择无缺失的基础特征）
    control_vars = ['X_r_ion', 'B_r_ion']
    partial_corr_results = []

    # 过滤掉名称含特殊字符的特征，避免计算错误
    valid_features = [f for f in feature_names if not any(c in f for c in ['(', ')', '/', '+', '-', '*'])]
    # 仅分析前20个基础特征
    top_features = valid_features[:20] if len(valid_features) > 20 else valid_features

    if len(top_features) == 0:
        print("无有效特征用于偏相关分析")
        # 返回空DataFrame
        return pd.DataFrame({
            'feature': [],
            'partial_corr': [],
            'p_value': [],
            'ci_95': []
        })

    for col in top_features:
        try:
            # 确保变量都在数据中
            if col not in df.columns or target_col not in df.columns:
                partial_corr_results.append({
                    'feature': col,
                    'partial_corr': np.nan,
                    'p_value': np.nan,
                    'ci_95': '变量不存在'
                })
                continue
            # 过滤缺失值
            temp_df = df[[col, target_col] + control_vars].dropna()
            if len(temp_df) < 10:
                partial_corr_results.append({
                    'feature': col,
                    'partial_corr': np.nan,
                    'p_value': np.nan,
                    'ci_95': '样本量不足'
                })
                continue
            # 计算偏相关系数
            pc = pg.partial_corr(data=temp_df, x=col, y=target_col, covar=control_vars)
            partial_corr_results.append({
                'feature': col,
                'partial_corr': pc['r'].values[0],
                'p_value': pc['p-val'].values[0],
                'ci_95': f"[{pc['CI95%'][0].split(',')[0][1:]:.4f}, {pc['CI95%'][0].split(',')[1][:-1]:.4f}]"
            })
        except Exception as e:
            partial_corr_results.append({
                'feature': col,
                'partial_corr': np.nan,
                'p_value': np.nan,
                'ci_95': f'计算失败：{str(e)[:50]}'
            })

    # 转换为DataFrame并排序
    pc_df = pd.DataFrame(partial_corr_results)
    pc_df = pc_df.sort_values('partial_corr', ascending=False, key=abs).reset_index(drop=True)

    # 保存结果
    pc_df.to_csv(os.path.join(OUTPUT_DIR, 'partial_corr_results.csv'), index=False, encoding='utf-8')

    # 可视化有效结果
    valid_pc = pc_df.dropna(subset=['partial_corr'])
    if len(valid_pc) > 0:
        top20 = valid_pc.head(20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='partial_corr', y='feature', data=top20, palette='magma')
        plt.title('偏相关系数（控制X_r_ion/B_r_ion，Top20特征）', fontsize=14)
        plt.xlabel('Partial Correlation')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'partial_corr_top20.png'), dpi=300)
        plt.close()

    # 输出统计信息
    valid_count = pc_df['partial_corr'].notna().sum()
    print(f"有效偏相关计算数量：{valid_count}/{len(top_features)}")
    if valid_count > 0:
        top_feature = pc_df.dropna().iloc[0]['feature']
        top_corr = pc_df.dropna().iloc[0]['partial_corr']
        print(f"最高偏相关特征：{top_feature} (r={top_corr:.4f})")
    else:
        print("偏相关分析无有效结果，跳过可视化")

    return pc_df


# ===================== 9. 方法6：SHAP Values分析（修复正则表达式问题） =====================
def shap_analysis(X_scaled_df, y, feature_names):
    """SHAP值分析（基于XGBoost模型）"""
    print("\n=== 方法6：SHAP值分析 ===")
    # 训练XGBoost模型
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_scaled_df, y)

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    # 计算特征的平均绝对SHAP值（全局重要性）
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_results = []

    for i, col in enumerate(feature_names):
        shap_results.append({
            'feature': col,
            'shap_importance': shap_importance[i]
        })

    # 转换为DataFrame并排序
    shap_df = pd.DataFrame(shap_results)
    shap_df = shap_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)

    # 保存结果
    shap_df.to_csv(os.path.join(OUTPUT_DIR, 'shap_results.csv'), index=False, encoding='utf-8')

    # 可视化SHAP总结图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled_df, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP特征重要性（Top20）', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=300)
    plt.close()

    # 可视化SHAP依赖图（前3个重要特征，跳过特殊字符特征）
    top3_features = shap_df.head(3)['feature'].tolist()
    for feat in top3_features:
        try:
            feat_idx = feature_names.index(feat)
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feat_idx, shap_values, X_scaled_df, feature_names=feature_names, show=False)
            # 清理特征名称作为文件名（修复正则表达式）
            clean_feat_name = clean_feature_name(feat)
            save_path = os.path.join(OUTPUT_DIR, f'shap_dependence_{clean_feat_name}.png')
            plt.title(f'SHAP依赖图 - {feat}', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"成功保存SHAP依赖图：{clean_feat_name}")
        except Exception as e:
            print(f"跳过SHAP依赖图保存（特征：{feat}）：{str(e)[:50]}")
            plt.close()

    # 输出统计信息
    print(f"SHAP最高重要性特征：{shap_df.iloc[0]['feature']} (SHAP值={shap_df.iloc[0]['shap_importance']:.4f})")

    return shap_df


# ===================== 10. 方法7：递归特征消除（RFE）- 稳定版 =====================
def rfe_analysis(X_scaled_df, y, feature_names):
    """递归特征消除（RFE）- 使用随机森林作为基模型"""
    print("\n=== 方法7：递归特征消除（RFE） ===")
    # 使用随机森林作为基模型
    estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # 选择Top20特征
    rfe = RFE(estimator=estimator, n_features_to_select=20, step=1)
    rfe.fit(X_scaled_df, y)

    # 整理结果
    rfe_results = []
    for i, col in enumerate(feature_names):
        rfe_results.append({
            'feature': col,
            'rfe_support': rfe.support_[i],
            'rfe_ranking': rfe.ranking_[i]
        })

    # 转换为DataFrame并排序
    rfe_df = pd.DataFrame(rfe_results)
    rfe_df = rfe_df.sort_values('rfe_ranking').reset_index(drop=True)

    # 保存结果
    rfe_df.to_csv(os.path.join(OUTPUT_DIR, 'rfe_results.csv'), index=False, encoding='utf-8')

    # 可视化选中的20个特征
    selected_features = rfe_df[rfe_df['rfe_support']]['feature'].tolist()
    if len(selected_features) > 0:
        selected_ranking = rfe_df[rfe_df['rfe_support']]['rfe_ranking'].tolist()
        plt.figure(figsize=(12, 8))
        sns.barplot(x=selected_ranking, y=selected_features, palette='tab10')
        plt.title('RFE选中的Top20特征（排名=1）', fontsize=14)
        plt.xlabel('RFE排名')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'rfe_selected_features.png'), dpi=300)
        plt.close()

    # 输出统计信息
    print(f"RFE选中的特征数量：{len(selected_features)}")
    if len(selected_features) > 0:
        print(f"RFE选中的前5个特征：{selected_features[:5]}")
    else:
        print("RFE未选中任何特征")

    return rfe_df


# ===================== 11. 方法8：方差膨胀因子（VIF）- 优化计算 =====================
def vif_analysis(X, feature_names):
    """方差膨胀因子（VIF）分析 - 经典多变量计算+高维稳定性处理"""
    print("\n=== 方法8：方差膨胀因子（VIF）分析 ===")

    # ===== 步骤1：预过滤高度相关特征，解决高维数值不稳定问题 =====
    # 计算特征间的皮尔逊相关系数
    corr_matrix = X.corr().abs()
    # 提取上三角矩阵（避免重复计算）
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # 找出相关系数>0.9的特征对
    high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
    # 保留低相关特征（用于VIF计算），同时记录所有特征的VIF结果
    low_corr_mask = ~X.columns.isin(high_corr_features)
    X_low_corr = X.loc[:, low_corr_mask]
    low_corr_feature_names = [name for idx, name in enumerate(feature_names) if low_corr_mask[idx]]

    print(f"高维预处理：过滤掉 {len(high_corr_features)} 个高度相关特征（|r|>0.9）")
    print(f"用于VIF计算的特征数量：{len(low_corr_feature_names)}")

    # ===== 步骤2：经典多变量VIF计算（仅对低相关特征） =====
    vif_results = []
    # 先初始化所有特征的VIF结果为NaN
    for name in feature_names:
        vif_results.append({
            'feature': name,
            'vif': np.nan,
            'high_collinearity': False,
            'status': '高相关过滤' if name in high_corr_features else '待计算'
        })

    # 对低相关特征计算VIF
    if len(X_low_corr) > 0 and X_low_corr.shape[1] > 0:
        for i, col in enumerate(low_corr_feature_names):
            try:
                # 经典多变量VIF计算（基于所有低相关特征的矩阵）
                vif = variance_inflation_factor(X_low_corr.values, i)
                # 处理数值溢出（VIF可能为inf）
                if np.isinf(vif) or vif > 1e10:
                    vif = 1e10  # 限制最大VIF值，避免可视化异常
                # 更新对应特征的VIF结果
                for res in vif_results:
                    if res['feature'] == col:
                        res['vif'] = vif
                        res['high_collinearity'] = vif >= THRESHOLDS['vif']
                        res['status'] = '计算完成'
                        break
            except Exception as e:
                for res in vif_results:
                    if res['feature'] == col:
                        res['status'] = f'计算失败：{str(e)[:20]}'
                        break

    # ===== 步骤3：整理结果并保存 =====
    vif_df = pd.DataFrame(vif_results)
    # 按VIF值降序排序（NaN排最后）
    vif_df = vif_df.sort_values(
        by=['vif', 'feature'],
        ascending=[False, True],
        na_position='last'
    ).reset_index(drop=True)

    # 保存详细结果（含状态说明）
    vif_df.to_csv(os.path.join(OUTPUT_DIR, 'vif_results.csv'), index=False, encoding='utf-8')

    # ===== 步骤4：可视化高共线性特征 =====
    # 仅可视化计算完成且VIF>5的特征
    high_vif = vif_df[
        (vif_df['high_collinearity']) &
        (vif_df['status'] == '计算完成')
        ].head(20)

    if len(high_vif) > 0:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='vif', y='feature', data=high_vif, palette='OrRd')
        plt.axvline(x=THRESHOLDS['vif'], color='green', linestyle='--', label=f'VIF阈值={THRESHOLDS["vif"]}')
        plt.title('高共线性特征（VIF>5，Top20）', fontsize=14)
        plt.xlabel('方差膨胀因子（VIF）')
        plt.ylabel('特征')
        plt.xscale('log')  # 对数刻度，适配VIF值差异大的情况
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'vif_high_collinearity.png'), dpi=300)
        plt.close()
    else:
        print("无高共线性特征（VIF>5）或无有效VIF计算结果")

    # ===== 步骤5：输出统计信息 =====
    # 统计有效计算的特征数
    valid_vif_count = len(vif_df[vif_df['status'] == '计算完成'])
    # 统计高共线性特征数
    high_collinearity_count = vif_df['high_collinearity'].sum()

    print(f"有效VIF计算数量：{valid_vif_count}/{len(feature_names)}")
    print(f"高共线性特征（VIF>={THRESHOLDS['vif']}）数量：{high_collinearity_count}/{valid_vif_count}")

    if valid_vif_count > 0:
        top_vif_row = vif_df[vif_df['status'] == '计算完成'].iloc[0]
        print(f"最高VIF特征：{top_vif_row['feature']} (VIF={top_vif_row['vif']:.2f})")

    return vif_df

# ===================== 12. 结果汇总 - 修复类型不匹配问题 =====================
def summarize_results(spearman_df, kendall_df, mi_df, dc_df, pc_df, shap_df, rfe_df, vif_df):
    """汇总所有方法的结果，修复索引类型不匹配问题"""
    print("\n=== 所有方法结果汇总 ===")
    # 以spearman_df的feature列为基准，合并所有结果
    summary_df = spearman_df[['feature', 'spearman_r', 'pass_threshold']].rename(
        columns={'pass_threshold': 'pass_spearman'})

    # 合并其他方法结果（使用merge避免索引问题）
    summary_df = summary_df.merge(
        kendall_df[['feature', 'kendall_tau']],
        on='feature', how='left'
    )
    summary_df = summary_df.merge(
        mi_df[['feature', 'mutual_info']],
        on='feature', how='left'
    )
    summary_df = summary_df.merge(
        dc_df[['feature', 'distance_corr']],
        on='feature', how='left'
    )
    summary_df = summary_df.merge(
        shap_df[['feature', 'shap_importance']],
        on='feature', how='left'
    )
    summary_df = summary_df.merge(
        rfe_df[['feature', 'rfe_support', 'rfe_ranking']],
        on='feature', how='left'
    )
    summary_df = summary_df.merge(
        vif_df[['feature', 'vif', 'high_collinearity']],
        on='feature', how='left'
    )

    # 计算各方法通过状态
    summary_df['pass_kendall'] = summary_df['kendall_tau'].abs() >= THRESHOLDS['kendall']
    summary_df['pass_mutual_info'] = summary_df['mutual_info'] >= THRESHOLDS['mutual_info']
    summary_df['pass_distance_corr'] = summary_df['distance_corr'] >= THRESHOLDS['distance_corr']
    summary_df['pass_vif'] = summary_df['vif'] < THRESHOLDS['vif']

    # 处理缺失值
    summary_df = summary_df.fillna({
        'kendall_tau': 0,
        'mutual_info': 0,
        'distance_corr': 0,
        'shap_importance': 0,
        'rfe_support': False,
        'vif': np.inf,
        'high_collinearity': True
    })

    # 计算综合得分（通过的方法数量）
    summary_df['total_pass'] = (
            summary_df['pass_spearman'].astype(int) +
            summary_df['pass_kendall'].astype(int) +
            summary_df['pass_mutual_info'].astype(int) +
            summary_df['pass_distance_corr'].astype(int) +
            summary_df['rfe_support'].astype(int) +
            summary_df['pass_vif'].astype(int)
    )

    # 按综合得分排序
    summary_df = summary_df.sort_values('total_pass', ascending=False).reset_index(drop=True)

    # 保存汇总表
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_analysis_summary.csv'), index=False, encoding='utf-8')

    # 可视化综合得分Top20
    top20_summary = summary_df.head(20)
    plt.figure(figsize=(14, 10))
    sns.barplot(x='total_pass', y='feature', data=top20_summary, palette='rainbow')
    plt.title('特征综合得分（通过方法数量，Top20）', fontsize=14)
    plt.xlabel('通过的分析方法数量（共6项）')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_summary_top20.png'), dpi=300)
    plt.close()

    # 输出最终筛选结果
    top_features = summary_df.head(20)['feature'].tolist()
    print(f"\n综合得分最高的20个核心特征：")
    for i, feat in enumerate(top_features, 1):
        score = summary_df.loc[summary_df['feature'] == feat, 'total_pass'].values[0]
        print(f"{i:2d}. {feat} (综合得分：{score})")

    return summary_df


# ===================== 13. 主函数：执行所有分析 =====================
def main():
    """主函数：执行所有8种特征分析方法"""
    # 加载数据
    df, X, X_scaled_df, y, feature_names = load_and_preprocess_data()

    # 执行各方法分析
    spearman_df = spearman_analysis(X, y, feature_names)
    kendall_df = kendall_analysis(X, y, feature_names)
    mi_df = mutual_info_analysis(X, y, feature_names)
    dc_df = distance_corr_analysis(X, y, feature_names)
    pc_df = partial_corr_analysis(df, feature_names, TARGET_COL)
    shap_df = shap_analysis(X_scaled_df, y, feature_names)
    rfe_df = rfe_analysis(X_scaled_df, y, feature_names)
    vif_df = vif_analysis(X, feature_names)

    # 汇总结果
    summary_df = summarize_results(
        spearman_df, kendall_df, mi_df, dc_df,
        pc_df, shap_df, rfe_df, vif_df
    )

    print("\n=== 所有分析完成！===")
    print(f"结果文件已保存至：{OUTPUT_DIR}")
    print(f"生成的文件包括：")
    print("  1. 各方法的详细结果CSV文件")
    print("  2. 特征相关性/重要性可视化图表")
    print("  3. 特征分析综合汇总表")


if __name__ == '__main__':
    main()