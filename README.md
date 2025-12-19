import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

# ====================== 1. 配置模块 ======================
def set_config():
    """设置全局配置（中文字体、图表样式）"""
    # 中文字体配置
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    # 图表样式配置
    plt.style.use('default')
    # 数据路径配置（改为你的实际路径）
    DATA_DIR = "F:\\多元设计\\raw\\"  # 重点修改这里
    return DATA_DIR

# ====================== 2. 数据加载模块 ======================
def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载所有CSV数据文件
    返回：字典{文件名: 数据框}
    """
    data_dict = {}
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"在{data_dir}目录下未找到CSV文件")
    
    # 逐个加载文件
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
            # 数据预处理：清理城市名称空格
            if '起点城市' in df.columns:
                df['起点城市'] = df['起点城市'].str.strip()
            data_dict[file] = df
            print(f"✅ 成功加载：{file}（{df.shape[0]}行 × {df.shape[1]}列）")
        except Exception as e:
            print(f"❌ 加载{file}失败：{str(e)}")
    
    return data_dict

# ====================== 3. 基础探索模块 ======================
def basic_data_explore(data_dict: Dict[str, pd.DataFrame]) -> None:
    """
    基础数据探索：输出文件结构、数据类型、缺失值情况
    """
    print("\n" + "="*60)
    print("📊 基础数据探索报告")
    print("="*60)
    
    # 分类整理文件（OD矩阵 vs 综合数据）
    od_files = [f for f in data_dict.keys() if 'od_matrix' in f]
    main_file = [f for f in data_dict.keys() if 'main_data' in f][0] if any('main_data' in f for f in data_dict.keys()) else None
    
    # 1. 分析OD矩阵文件
    print("\n1. OD矩阵文件分析（城市间交互数据）：")
    for file in od_files:
        df = data_dict[file]
        print(f"\n📄 {file}：")
        print(f"   列名：{', '.join(df.columns)}")
        print(f"   数据类型：\n{df.dtypes.to_string()}")
        print(f"   缺失值比例：{(df.isnull().sum()/len(df)*100).round(2).to_string()}%")
    
    # 2. 分析综合数据文件
    if main_file:
        df_main = data_dict[main_file]
        print(f"\n2. 综合数据文件分析（{main_file}）：")
        print(f"   时间范围：{sorted(df_main['年份'].unique())}")
        print(f"   覆盖城市：{sorted(df_main['城市'].unique())}")
        print(f"   核心指标分类：")
        # 指标分类（基于字段名关键词）
        indicator_categories = {
            '数据产业': [col for col in df_main.columns if any(key in col for key in ['数据', 'API', '带宽', '算力', '机架'])],
            '经济发展': [col for col in df_main.columns if any(key in col for key in ['GDP', '数字经济', '外贸', 'FDI', '电商'])],
            '科技创新': [col for col in df_main.columns if any(key in col for key in ['研发', '专利', '高新', '科技型', '独角兽'])],
            '基础设施': [col for col in df_main.columns if any(key in col for key in ['5G', '基站', '光网', '物联网'])]
        }
        for cate, cols in indicator_categories.items():
            if cols:
                print(f"      - {cate}（{len(cols)}个）：{', '.join(cols[:3])}...")

# ====================== 4. 深度分析模块 ======================
def deep_data_analysis(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    深度分析：计算关键指标（年度增长、城市排名、相关性）
    返回：分析结果字典
    """
    print("\n" + "="*60)
    print("🔍 深度数据分析报告")
    print("="*60)
    
    # 1. 提取核心数据
    od_summary = data_dict['od_matrix.csv']  # 汇总OD矩阵
    main_data = data_dict['main_data_advanced.csv']  # 综合数据
    yearly_od_files = [f for f in data_dict.keys() if 'od_matrix_20' in f]  # 年度OD矩阵
    
    # 2. 计算年度数据传输增长
    yearly_transfer = od_summary.groupby('年份')['数据传输量_TB'].sum().reset_index()
    growth_rate = ((yearly_transfer.iloc[-1]['数据传输量_TB'] - yearly_transfer.iloc[0]['数据传输量_TB']) / 
                   yearly_transfer.iloc[0]['数据传输量_TB'] * 100)
    print(f"\n1. 年度数据传输量增长分析：")
    print(f"   2019年总量：{yearly_transfer.iloc[0]['数据传输量_TB']:.0f} TB")
    print(f"   2023年总量：{yearly_transfer.iloc[-1]['数据传输量_TB']:.0f} TB")
    print(f"   五年增长率：{growth_rate:.1f}%")
    
    # 3. 2023年城市间交互排名
    transfer_2023 = od_summary[od_summary['年份'] == 2023]
    top10_transfer = transfer_2023.nlargest(10, '数据传输量_TB')
    print(f"\n2. 2023年数据传输量Top5城市对：")
    for i, (_, row) in enumerate(top10_transfer.head(5).iterrows(), 1):
        print(f"   {i}. {row['起点城市']}→{row['终点城市']}：{row['数据传输量_TB']:.0f} TB")
    
    # 4. 核心城市数字经济水平
    main_2023 = main_data[main_data['年份'] == 2023]
    core_cities = ['广州', '深圳', '香港', '澳门']
    core_econ = main_2023[main_2023['城市'].isin(core_cities)][['城市', 'GDP_亿元', '数字经济占GDP比重_%']]
    print(f"\n3. 2023年核心城市数字经济水平：")
    print(core_econ.sort_values('数字经济占GDP比重_%', ascending=False).to_string(index=False))
    
    # 返回分析结果
    analysis_results = {
        'yearly_transfer': yearly_transfer,
        'top10_transfer_2023': top10_transfer,
        'core_city_econ_2023': core_econ,
        'main_2023': main_2023
    }
    return analysis_results

# ====================== 5. 可视化模块 ======================
def create_visualizations(analysis_results: Dict[str, pd.DataFrame], save_path: str = '/mnt/') -> None:
    """
    生成4个核心可视化图表：趋势图、排名图、对比图、关系图
    """
    print("\n" + "="*60)
    print("🎨 生成可视化图表")
    print("="*60)
    
    # 提取分析结果
    yearly_transfer = analysis_results['yearly_transfer']
    top10_transfer = analysis_results['top10_transfer_2023']
    core_econ = analysis_results['core_city_econ_2023']
    main_2023 = analysis_results['main_2023']
    
    # 创建2×2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Greater Bay Area Digital Economy Analysis (2019-2023)', fontsize=16, fontweight='bold')
    
    # 1. 图1：年度数据传输量趋势（左上）
    axes[0,0].plot(yearly_transfer['年份'], yearly_transfer['数据传输量_TB'], 
                   marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    axes[0,0].fill_between(yearly_transfer['年份'], yearly_transfer['数据传输量_TB'], 
                           alpha=0.3, color='#2E86AB')
    axes[0,0].set_title('Total Data Transfer Volume (2019-2023)', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Data Transfer (TB)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(yearly_transfer['年份'])
    
    # 2. 图2：2023年Top10城市对（右上）
    top10_transfer['city_pair'] = top10_transfer['起点城市'] + '→' + top10_transfer['终点城市']
    bars = axes[0,1].barh(range(len(top10_transfer)), top10_transfer['数据传输量_TB'], 
                          color='#A23B72')
    axes[0,1].set_yticks(range(len(top10_transfer)))
    axes[0,1].set_yticklabels(top10_transfer['city_pair'], fontsize=10)
    axes[0,1].set_title('Top 10 City Pairs by Data Transfer (2023)', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Data Transfer (TB)')
    axes[0,1].grid(True, alpha=0.3, axis='x')
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0,1].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.0f}', ha='left', va='center', fontsize=9)
    
    # 3. 图3：核心城市数字经济占比（左下）
    cities = core_econ['城市'].tolist()
    digital_ratios = core_econ['数字经济占GDP比重_%'].tolist()
    bars3 = axes[1,0].bar(cities, digital_ratios, color=['#F18F01', '#C73E1D', '#2E86AB', '#A23B72'])
    axes[1,0].set_title('Digital Economy Ratio in Core Cities (2023)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('City')
    axes[1,0].set_ylabel('Digital Economy / GDP (%)')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, ratio in zip(bars3, digital_ratios):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                       f'{ratio:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. 图4：数据中心与算力关系（右下）
    scatter = axes[1,1].scatter(main_2023['数据中心数量'], main_2023['算力规模_PFLOPS'], 
                               s=200, alpha=0.6, c=range(len(main_2023)), cmap='viridis')
    # 添加城市标签
    for _, row in main_2023.iterrows():
        axes[1,1].annotate(row['城市'], (row['数据中心数量'], row['算力规模_PFLOPS']),
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1,1].set_title('Data Centers vs Computing Power (2023)', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Number of Data Centers')
    axes[1,1].set_ylabel('Computing Power (PFLOPS)')
    axes[1,1].grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    save_file = os.path.join(save_path, 'gba_digital_economy_analysis.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存至：{save_file}")

# ====================== 6. 主函数（流程控制） ======================
def main():
    try:
        # 1. 初始化配置
        DATA_DIR = set_config()
        print("🔧 初始化完成，开始数据处理...")
        
        # 2. 加载数据
        data_dict = load_all_data(DATA_DIR)
        
        # 3. 基础探索
        basic_data_explore(data_dict)
        
        # 4. 深度分析
        analysis_results = deep_data_analysis(data_dict)
        
        # 5. 生成可视化
        create_visualizations(analysis_results, DATA_DIR)
        
        print("\n" + "="*60)
        print("🎉 数据分析完成！生成文件：")
        print(f"   1. 可视化图表：gba_digital_economy_analysis.png")
        print(f"   2. 分析报告：控制台输出（可复制保存）")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")

# 执行主函数
if __name__ == "__main__":
    main()
