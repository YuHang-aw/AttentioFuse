import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple, Optional

# 科研配色方案 - 更清晰的颜色
OMICS_COLORS = {
    'mrna': '#EF767A',    # 新红色
    'CNV': '#456990',     # 深蓝色
    'SNV': '#48C0AA',     # 青绿色
    'fusion': '#FFA500'    # 橙色
}

def process_multiomics_file(file_path: str) -> pd.DataFrame:
    """处理多组学数据文件"""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['omics', 'layer', 'node_index', 'importance', 'mapped_name']
        
        # 检查必需的列
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 文件 {file_path} 缺少列 '{col}'")
                return None
        
        # 将NaN填充为空字符串
        if 'mapped_name' in df.columns:
            df['mapped_name'] = df['mapped_name'].fillna('')
        
        # 过滤掉未映射的节点
        df = df[df['mapped_name'] != '']
        
        # 为fusion层节点添加特殊标记
        df['is_fusion'] = df['layer'] == 'fusion'
        
        return df
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def get_top_features(df: pd.DataFrame, omics_type: str, layer: str, top_n: int = 10) -> pd.DataFrame:
    """获取特定组学类型和层的前N个重要特征"""
    # 对于fusion层，可以减少选取的特征数量
    if layer == 'fusion':
        top_n = 5  # fusion层只选取Top5
    
    # 过滤指定组学和层
    subset = df[(df['omics'] == omics_type) & (df['layer'] == layer)]
    
    # 按重要性降序排列
    subset = subset.sort_values('importance', ascending=False)
    
    # 返回前N个特征
    return subset.head(top_n)


def create_sankey_data(df: pd.DataFrame, omics_type: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """创建桑基图的节点和链接数据（增强版）"""
    # 新增数据预处理
    df = df.copy()
    df['layer'] = df['layer'].astype(str)
    
    # 处理fusion层特殊逻辑
    fusion_nodes = df[df['layer'] == 'fusion'].copy()
    if not fusion_nodes.empty:
        # 为fusion节点生成唯一标识
        fusion_nodes['node_id'] = fusion_nodes.apply(
            lambda x: f"fusion_{x['source_omics']}_{x['node_index']}", axis=1
        )
        df.update(fusion_nodes)
    
    # 确保所有节点都有有效颜色映射
    color_mapping = {
        'mrna': '#EF767A',
        'CNV': '#456990',
        'SNV': '#48C0AA',
        'fusion': '#FFA500'
    }
    df['color'] = df['omics'].map(color_mapping).fillna('#888888')
    
    # 重新设计层连接逻辑
    layers_order = ['input', 'layer_1', 'layer_2', 'layer_3', 'fusion']
    valid_layers = [l for l in layers_order if l in df['layer'].unique()]
    
    # 生成节点列表（增强处理）
    nodes = []
    node_id_map = {}
    for layer in valid_layers:
        layer_data = df[df['layer'] == layer]
        for _, row in layer_data.iterrows():
            node_id = f"{row['omics']}_{layer}_{row['node_index']}"
            if node_id not in node_id_map:
                node_id_map[node_id] = len(nodes)
                nodes.append({
                    'id': node_id,
                    'name': row['mapped_name'],
                    'layer': layer,
                    'omics': row['omics'],
                    'color': row['color'],
                    'importance': row['importance']
                })
    
    # 生成链接逻辑（关键修改）
    links = []
    for i in range(len(valid_layers)-1):
        current_layer = valid_layers[i]
        next_layer = valid_layers[i+1]
        
        current_data = df[df['layer'] == current_layer]
        next_data = df[df['layer'] == next_layer]
        
        # 处理常规层间连接
        if next_layer != 'fusion':
            for _, src_row in current_data.iterrows():
                src_id = f"{src_row['omics']}_{current_layer}_{src_row['node_index']}"
                src_idx = node_id_map[src_id]
                
                # 连接同组学的下一层节点
                same_omics_next = next_data[next_data['omics'] == src_row['omics']]
                for _, tgt_row in same_omics_next.iterrows():
                    tgt_id = f"{tgt_row['omics']}_{next_layer}_{tgt_row['node_index']}"
                    tgt_idx = node_id_map[tgt_id]
                    
                    links.append({
                        'source': src_idx,
                        'target': tgt_idx,
                        'value': (src_row['importance'] + tgt_row['importance'])/2,
                        'color': src_row['color']
                    })
        
        # 处理到fusion层的特殊连接
        else:
            for _, src_row in current_data.iterrows():
                src_id = f"{src_row['omics']}_{current_layer}_{src_row['node_index']}"
                src_idx = node_id_map[src_id]
                
                # 连接到所有fusion节点（加权连接）
                for _, tgt_row in next_data.iterrows():
                    tgt_id = f"{tgt_row['omics']}_{next_layer}_{tgt_row['node_index']}"
                    tgt_idx = node_id_map[tgt_id]
                    
                    # 计算连接强度（基于来源组学相关性）
                    weight = 1.0 if tgt_row['source_omics'] == src_row['omics'] else 0.3
                    links.append({
                        'source': src_idx,
                        'target': tgt_idx,
                        'value': (src_row['importance'] * weight + tgt_row['importance'])/2,
                        'color': src_row['color']
                    })
    
    return pd.DataFrame(nodes), pd.DataFrame(links)

def export_top_molecules_pathways(nodes: pd.DataFrame, links: pd.DataFrame, output_path: str):
    """
    导出顶级分子和连接的通路到CSV文件
    
    参数:
        nodes: 节点DataFrame
        links: 链接DataFrame
        output_path: 输出文件路径（不含扩展名）
    """
    # 1. 将links转换为更易读的格式
    pathway_connections = []
    
    for _, link in links.iterrows():
        source_idx = int(link['source'])
        target_idx = int(link['target'])
        
        # 获取源节点和目标节点的信息
        source_node = nodes.iloc[source_idx]
        target_node = nodes.iloc[target_idx]
        
        pathway_connections.append({
            'Source_Molecule': source_node['name'],
            'Source_Omics': source_node['omics'],
            'Source_Layer': source_node['layer'],
            'Source_Importance': source_node['importance'],
            'Target_Molecule': target_node['name'],
            'Target_Omics': target_node['omics'],
            'Target_Layer': target_node['layer'],
            'Connection_Strength': link['value']
        })
    
    # 2. 保存为CSV
    connections_df = pd.DataFrame(pathway_connections)
    
    # 3. 按源节点重要性降序排序
    connections_df = connections_df.sort_values('Source_Importance', ascending=False)
    
    # 4. 保存到CSV文件
    csv_output_path = f"{output_path[:-5]}_pathways.csv"  # 替换.html为_pathways.csv
    connections_df.to_csv(csv_output_path, index=False)
    
    print(f"分子通路关系已导出至: {csv_output_path}")
    
    # 5. 创建单独的节点列表CSV，按重要性排序
    top_nodes = nodes.sort_values('importance', ascending=False)
    nodes_csv_path = f"{output_path[:-5]}_nodes.csv"
    top_nodes.to_csv(nodes_csv_path, index=False)
    
    print(f"节点列表已导出至: {nodes_csv_path}")

def generate_sankey_diagram(nodes: pd.DataFrame, links: pd.DataFrame, title: str, output_path: str):
    """生成桑基图并保存（修复版）"""
    # 节点颜色处理
    node_colors = []
    for _, row in nodes.iterrows():
        omics = row['omics']
        layer = row['layer']
        
        # 强制转换组学类型
        if layer == 'fusion':
            color = OMICS_COLORS['fusion']
        else:
            # 处理可能的组学类型错误
            if omics not in OMICS_COLORS:
                print(f"警告: 未知组学类型 '{omics}'，使用默认灰色")
                color = 'rgba(150,150,150,0.5)'
            else:
                color = OMICS_COLORS[omics]
        node_colors.append(color)
    
    # 链接颜色处理（关键修复点）
    link_colors = []
    for _, link in links.iterrows():
        source_idx = link['source']
        source_omics = nodes.iloc[source_idx]['omics']
        
        # 获取基础颜色
        if source_omics == 'fusion' or nodes.iloc[source_idx]['layer'] == 'fusion':
            base_color = OMICS_COLORS['fusion']
        else:
            base_color = OMICS_COLORS.get(source_omics, 'rgba(150,150,150,0.3)')
        
        # 转换为RGBA格式并调整透明度
        if base_color.startswith('#'):
            rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            link_color = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)'  # 统一透明度
        else:
            # 处理已有rgba颜色
            parts = base_color[5:-1].split(',')
            link_color = f'rgba({parts[0]},{parts[1]},{parts[2]},0.2)'
        
        link_colors.append(link_color)
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad=30,
            thickness=20,
            line=dict(color="black", width=0.3),
            label=nodes['name'].tolist(),
            color=node_colors,
            hovertemplate='<b>%{label}</b><br>Importance: %{value:.2f}<extra></extra>'
        ),
        link = dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=link_colors  # 使用统一处理后的颜色
        )
    )])
    
    # 优化布局
    fig.update_layout(
        title_text=title,
        font=dict(size=10, family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=1400,
        width=1800,
        margin=dict(l=50, r=300, t=100, b=100),
        hovermode='x'
    )
    
    # 添加动态图例
    fig.add_annotation(
        x=1.1,
        y=0.9,
        xref="paper",
        yref="paper",
        showarrow=False,
        text="<span style='color:#EF767A'>■ Transcriptome</span><br>" +
             "<span style='color:#456990'>■ CNV</span><br>" +
             "<span style='color:#48C0AA'>■ SNV</span><br>",
        align="left",
        bordercolor="#666",
        borderwidth=1
    )
    
    # 保存文件
    fig.write_image(output_path.replace(".html", ".pdf"), scale=2, engine="kaleido")
    fig.write_html(output_path)

def merge_pdfs(output_dir):
    from PyPDF2 import PdfMerger
    
    merger = PdfMerger()
    for suffix in ['t', 'm', 'n']:
        pdf_path = os.path.join(output_dir, 
            f'multi_omics_fusion_att_importance_scores_{suffix}_processed_all_omics_fusion.pdf')
        if os.path.exists(pdf_path):
            merger.append(pdf_path)
    
    merged_path = os.path.join(output_dir, 'combined_multi_omics_fusion.pdf')
    merger.write(merged_path)
    merger.close()
    print(f'合并完成：{merged_path}')

def generate_combined_sankey(input_dir, output_dir):
    """分步生成并合并PDF的最终解决方案"""
    import tempfile
    from PyPDF2 import PdfMerger
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_paths = []
        
        for suffix in ['t', 'm', 'n']:
            # 生成单个PDF
            file_name = f'multi_omics_fusion_att_importance_scores_{suffix}_processed.csv'
            file_path = os.path.join(input_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"跳过缺失文件: {file_path}")
                continue
                
            try:
                # 处理数据
                df = process_multiomics_file(file_path)
                if df is None or df.empty:
                    continue
                
                # 生成桑基数据
                nodes, links = create_sankey_data(df)
                
                # 创建独立图表
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=15,
                        line=dict(width=0.3),
                        label=nodes['name'],
                        color=nodes['color'],
                        hovertemplate='<b>%{label}</b><br>Importance: %{value:.2f}<extra></extra>'
                    ),
                    link=dict(
                        source=links['source'],
                        target=links['target'],
                        value=links['value'],
                        color=links['color']
                    )
                ))
                
                # 优化布局参数
                fig.update_layout(
                    title_text=f"{suffix.upper()} Stage Features",
                    height=900,  # 适当降低高度
                    width=1800,
                    margin=dict(l=50, r=300, t=80, b=50),
                    font=dict(size=10)
                )
                
                # 生成临时PDF
                temp_pdf = os.path.join(temp_dir, f"temp_{suffix}.pdf")
                fig.write_image(temp_pdf, scale=1.5)  # 降低缩放系数
                pdf_paths.append(temp_pdf)
                
            except Exception as e:
                print(f"生成{suffix}阶段图表时出错: {e}")
                continue
        
        # 合并PDF
        if pdf_paths:
            merger = PdfMerger()
            for pdf in pdf_paths:
                merger.append(pdf)
            
            output_path = os.path.join(output_dir, 'combined_vertical_fusion.pdf')
            merger.write(output_path)
            merger.close()
            print(f"成功生成合并PDF: {output_path}")
        else:
            print("警告: 没有可合并的PDF文件")

def process_all_files(input_dir: str, output_dir: str):
    """处理目录中的所有多组学数据文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for file_name in csv_files:
        file_path = os.path.join(input_dir, file_name)
        print(f"\n处理文件: {file_name}")
        
        # 处理文件
        df = process_multiomics_file(file_path)
        if df is None:
            continue
        
        # 获取所有组学类型（排除fusion）
        omics_types = [o for o in df['omics'].unique() if o != 'fusion']
        
        # 为每个组学类型生成Top10桑基图
        for omics in omics_types:
            print(f"  生成 {omics} 组学的Top10桑基图...")
            
            # 为每层获取Top10特征
            top_features = pd.DataFrame()
            layers = sorted([l for l in df['layer'].unique() if l != 'fusion'])
            
            for layer in layers:
                top_layer_features = get_top_features(df, omics, layer)
                top_features = pd.concat([top_features, top_layer_features])
            
            # 添加相应的融合层节点 (只取Top5)
            fusion_nodes = df[df['layer'] == 'fusion'].sort_values('importance', ascending=False).head(5)
            if not fusion_nodes.empty:
                top_features = pd.concat([top_features, fusion_nodes])
            
            # 创建桑基图数据
            nodes, links = create_sankey_data(top_features, omics)
            
            if len(nodes) > 0 and len(links) > 0:
                # 生成桑基图
                output_file = os.path.join(output_dir, f"{file_name[:-4]}_{omics}_top10.html")
                generate_sankey_diagram(nodes, links, f"{omics} Top10 Features", output_file)
            else:
                print(f"  警告: {omics} 组学没有足够的数据生成桑基图")
        
        # 生成融合所有组学的桑基图
        print("  生成融合所有组学的桑基图...")
        
        # 为每个组学和每层获取特征
        all_top_features = pd.DataFrame()
        
        for omics in omics_types:
            for layer in sorted([l for l in df['layer'].unique() if l != 'fusion']):
                layer_features = get_top_features(df, omics, layer, top_n=5)  # 每个组学每层选取Top5
                all_top_features = pd.concat([all_top_features, layer_features])
        
        # 添加融合层 (只选Top5)
        fusion_nodes = df[df['layer'] == 'fusion']
        fusion_top = fusion_nodes.sort_values('importance', ascending=False).head(5)
        all_top_features = pd.concat([all_top_features, fusion_top])
        
        # 创建桑基图数据
        nodes, links = create_sankey_data(all_top_features)
        
        if len(nodes) > 0 and len(links) > 0:
            # 生成桑基图
            output_file = os.path.join(output_dir, f"{file_name[:-4]}_all_omics_fusion.html")
            generate_sankey_diagram(nodes, links, "Multi-omics Integration Features", output_file)
        else:
            print("  警告: 没有足够的数据生成融合桑基图")
        #merge_pdfs(output_dir)
        generate_combined_sankey(input_dir, output_dir)

# 主函数
def main():
    # 设置输入和输出目录
    input_dir = "lusc_att_interpre"  # 包含CSV文件的目录
    output_dir = "lusc_att_interpre/sankey_diagrams_3"  # 输出桑基图的目录
    
    process_all_files(input_dir, output_dir)

if __name__ == "__main__":
    main()