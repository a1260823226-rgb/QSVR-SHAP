import pandas as pd
import numpy as np
import os

# ===================== ATTENTION!本地路径配置 =====================
input_path = r'YOUR FILE_PATH'
output_path = r'YOUR FILE_PATH'

# ===================== 数据清洗核心逻辑=====================
# 1. 加载原始数据集并查看基本信息
print("=== 步骤1：加载原始数据集 ===")
# 检查原始文件是否存在，避免路径错误
if os.path.exists(input_path):
    df = pd.read_csv(input_path)
    print(f"原始数据集形状：{df.shape} (样本数 × 特征数)")
    print(f"原始数据集列数：{len(df.columns)}")
    print(f"原始数据集中带隙为0的样本数：{len(df[df['bandgap'] == 0])}")
else:
    print(f"错误：原始文件不存在！请检查路径是否正确：\n{input_path}")
    exit()  # 路径错误时终止程序，避免后续报错

# 2. 清洗步骤1：删除无意义的num标识列
print("\n=== 步骤2：删除冗余的num列 ===")
if 'num' in df.columns:
    df = df.drop('num', axis=1)
    print("已删除num列（样本编号，无物理意义）")
else:
    print("num列不存在，无需删除")
print(f"删除后数据集形状：{df.shape}")

# 3. 清洗步骤2：处理带隙为0的异常样本（1%分位数替换）
print("\n=== 步骤3：处理带隙为0的异常样本 ===")
# 计算带隙的1%分位数（仅基于非0带隙样本，避免被异常值干扰）
valid_bandgap = df[df['bandgap'] > 0]['bandgap']
if len(valid_bandgap) > 0:  # 确保有有效带隙样本
    bandgap_1percentile = valid_bandgap.quantile(0.01)
    print(f"带隙1%分位数：{bandgap_1percentile:.4f} eV（用于替换0值）")

    # 替换带隙为0的样本
    zero_bandgap_count = len(df[df['bandgap'] == 0])
    df.loc[df['bandgap'] == 0, 'bandgap'] = bandgap_1percentile
    print(f"已将{zero_bandgap_count}个带隙为0的样本替换为{bandgap_1percentile:.4f} eV")
    print(f"处理后带隙范围：{df['bandgap'].min():.4f} - {df['bandgap'].max():.4f} eV")
else:
    print("警告：所有样本带隙均为0，无法计算分位数，跳过替换步骤")

# 4. 清洗步骤3：优化元素分类列的数据类型（转换为category）
print("\n=== 步骤4：优化元素分类列的数据类型 ===")
category_columns = ['A_element', 'B_element', 'X_elements']
# 从structure列提取元素信息（若未提前添加）
if 'A_element' not in df.columns:
    def extract_elements_from_structure(structure_str):
        base_str = structure_str.replace('_POSCAR', '')
        elements = base_str.split('_')
        if len(elements) >= 4:  # 确保结构格式正确（如Cs_Ba_Br_Br_Br）
            A_element = elements[0]
            B_element = elements[1]
            X_elements = ','.join(elements[2:])  # 合并多个X位卤素（如Br,Br,Br）
            return A_element, B_element, X_elements
        else:
            return 'Unknown', 'Unknown', 'Unknown'  # 处理异常结构格式


    structure_analysis = df['structure'].apply(extract_elements_from_structure)
    df[['A_element', 'B_element', 'X_elements']] = pd.DataFrame(
        structure_analysis.tolist(), index=df.index
    )
    print("已从structure列提取A_element、B_element、X_elements列")

# 转换为category类型（优化内存和后续编码效率）
for col in category_columns:
    if col in df.columns:
        original_dtype = df[col].dtype
        df[col] = df[col].astype('category')
        print(f"已将{col}列从{original_dtype}转换为category类型")
    else:
        print(f"警告：{col}列不存在，跳过类型转换")

# 5. 验证清洗结果（确保数据质量）
print("\n=== 步骤5：验证清洗后的数据质量 ===")
print(f"最终数据集形状：{df.shape} (样本数 × 特征数)")
# 检查缺失值
missing_max = df.isnull().sum().max()
print(f"缺失值检查：所有列缺失值数量均为 {missing_max}（0表示无缺失）")
# 检查分类特征和目标变量
category_features = [col for col in df.columns if df[col].dtype.name == 'category']
print(f"数据类型检查：")
print(f"  - 分类特征（category）：{category_features if category_features else '无'}")
if 'bandgap' in df.columns:
    print(f"  - 目标变量（bandgap）：{df['bandgap'].dtype}，均值 {df['bandgap'].mean():.4f} eV")
else:
    print("警告：目标变量bandgap列不存在！")

# 6. 保存清洗后的数据集（本地路径）
print("\n=== 步骤6：保存清洗后的数据集 ===")
try:
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"清洗后的数据集已成功保存至：\n{output_path}")

    # 对比文件大小（验证内存优化效果）
    original_size = os.path.getsize(input_path) / 1024  # 转换为KB
    cleaned_size = os.path.getsize(output_path) / 1024
    size_change = ((original_size - cleaned_size) / original_size) * 100
    print(f"文件大小对比：")
    print(f"  - 原始文件：{original_size:.2f} KB")
    print(f"  - 清洗后文件：{cleaned_size:.2f} KB")
    print(f"  - 内存变化比例：{size_change:.1f}%（负值表示因新增列略有增大，属正常）")
    print("\n=== 数据清洗任务全部完成！===")
except Exception as e:
    print(f"保存文件失败！错误原因：{str(e)}")

    print("请检查目标文件夹是否有权限写入（如管理员权限）")
