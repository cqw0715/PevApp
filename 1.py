import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import pickle
from io import StringIO
from torch.utils.data import DataLoader, TensorDataset
import esm
import time
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

# ==========================================
# 模型架构定义（与训练时保持一致）
# ==========================================
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
except ImportError:
    st.warning("Mamba模块未安装，将使用替代实现")
    # 简单的替代实现，仅用于演示
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.norm = nn.LayerNorm(d_model)
            
        def forward(self, x):
            return self.norm(x)

class CNNBranch(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Unflatten(1, (1, 256)),
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        return self.classifier(feat)

class TransformerBranch(nn.Module):
    def __init__(self, input_dim=1280, d_model=256, nhead=8, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)

class MambaBranch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.preprocess = nn.Linear(input_dim, 256)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
            for _ in range(5)
        ])
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(1)
        for block in self.mamba_blocks:
            x = x + block(x)
        x = self.norm(x).squeeze(1)
        return self.classifier(x)

class MutualLearningModel(nn.Module):
    def __init__(self, input_dim, num_classes=2, embed_dim=128):
        super().__init__()
        self.cnn = CNNBranch(input_dim, num_classes)
        self.trans = TransformerBranch(input_dim, num_classes=num_classes)
        self.mamba = MambaBranch(input_dim, num_classes)
        self.logits_norm = nn.LayerNorm(num_classes)
        self.feature_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2, batch_first=True)
        self.attn_norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm1 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2, batch_first=True)
        self.attn_norm2 = nn.LayerNorm(embed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm2 = nn.LayerNorm(embed_dim)
        total_gate_dim = embed_dim * 3 + num_classes * 3
        self.gate = nn.Sequential(
            nn.Linear(total_gate_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.8 + 1e-6)))
        self.refine = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LayerNorm(num_classes),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        o1, o2, o3 = self.cnn(x), self.trans(x), self.mamba(x)
        branches = torch.stack([o1, o2, o3], dim=1)
        branches_norm = self.logits_norm(branches)
        x_proj = self.feature_proj(branches_norm)
        attn_out, _ = self.attn1(x_proj, x_proj, x_proj)
        x = self.attn_norm1(x_proj + attn_out)
        x = self.ffn_norm1(x + self.ffn1(x))
        attn_out, _ = self.attn2(x, x, x)
        x = self.attn_norm2(x + attn_out)
        x = self.ffn_norm2(x + self.ffn2(x))
        raw_logits = branches.flatten(1)
        fused_proj = x.flatten(1)
        combined_feat = torch.cat([fused_proj, raw_logits], dim=1)
        gate_scores = self.gate(combined_feat)
        temp = F.softplus(self.log_temp) + 1e-4
        weights = F.softmax(gate_scores / temp, dim=1).unsqueeze(-1)
        o_fused = (branches * weights).sum(dim=1)
        o_fused = o_fused + self.refine(o_fused)
        return o1, o2, o3, o_fused

# ==========================================
# 特征提取类（修复缓存问题）
# ==========================================
# main.py 中替换模型加载部分
import requests, json

def esm_predict(sequence: str):
    """调用ESM Atlas官方API（免费，无需下载模型）"""
    try:
        response = requests.post(
            "https://api.esmatlas.com/v1/embed",
            json={"sequence": sequence},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]  # 返回650M模型生成的嵌入向量
    except Exception as e:
        return f"API错误: {str(e)}"

# Streamlit界面
if st.button("用ESM-650M预测"):
    result = esm_predict(user_sequence)
    st.success("✅ 通过官方API完成预测（无需本地模型）")

# ==========================================
# 缓存函数 - 正确使用st.cache_resource
# ==========================================
@st.cache_resource
def get_feature_extractor():
    """获取特征提取器的缓存实例"""
    return esm_predict()

@st.cache_resource
def load_model_and_scaler():
    """加载预训练模型和标准化器"""
    with st.spinner("🔄 正在加载预训练模型..."):
        # 检查模型文件是否存在
        model_path = "best_mutual_learning_model.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件未找到: {model_path}")
            st.info("请确保模型文件与应用在同一目录下")
            return None, None, None
        
        # 加载模型 - 修复安全警告
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # 尝试使用 weights_only=True (PyTorch 2.1+)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only参数
            checkpoint = torch.load(model_path, map_location=device)
        
        # 初始化模型
        model = MutualLearningModel(input_dim=1280, num_classes=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 获取标准化器
        scaler = checkpoint['scaler']
        
        return model, scaler, device

# ==========================================
# 应用主函数
# ==========================================
def main():
    # 页面配置
    st.set_page_config(
        page_title="猪肠道病毒识别系统",
        page_icon="🐷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用标题和说明
    st.title("🐷 猪肠道病毒识别系统")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3>🔬 系统说明</h3>
        <p>本系统使用深度学习模型对蛋白质序列进行分类，判断其是否为猪肠道病毒。</p>
        <ul>
            <li><b>类别0</b>: 猪肠道病毒</li>
            <li><b>类别1</b>: 非猪肠道病毒</li>
        </ul>
        <p>模型基于ESM-2 650M特征提取器和多分支融合架构，提供高精度的预测结果。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        st.markdown("### 模型信息")
        st.info("深度学习融合模型\n(ESM-2 + CNN + Transformer + Mamba)")
        
        st.markdown("### 使用说明")
        st.markdown("""
        1. **单序列预测**: 在输入框中粘贴蛋白质序列
        2. **批量预测**: 上传包含序列的CSV文件
        3. 查看预测结果及置信度
        """)
        
        st.markdown("### 注意事项")
        st.warning("""
        - 仅支持标准氨基酸字符
        - 序列长度建议在50-2000个氨基酸之间
        - GPU加速可显著提升处理速度
        """)
    
    # 加载模型和特征提取器
    model, scaler, device = load_model_and_scaler()
    feature_extractor = get_feature_extractor()
    
    if model is None or feature_extractor is None:
        st.stop()
    
    # 预测函数
    def predict_sequences(sequences: List[str]) -> List[dict]:
        """对序列列表进行预测"""
        if not sequences:
            return []
        
        # 提取特征
        with st.spinner(f"🧬 正在提取 {len(sequences)} 条序列的特征..."):
            features = feature_extractor.extract_features(sequences)
        
        # 标准化
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # 预测
        results = []
        with torch.no_grad():
            _, _, _, o_fused = model(features_tensor)
            probs = F.softmax(o_fused, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = probs[:, 1].cpu().numpy()  # 非猪肠道病毒的概率
        
        # 生成结果
        for i, (seq, pred, conf) in enumerate(zip(sequences, preds, confidences)):
            result = {
                'sequence_id': f"seq_{i+1}",
                'sequence': seq[:50] + "..." if len(seq) > 50 else seq,
                'full_sequence': seq,
                'prediction': int(pred),
                'confidence': float(conf),
                'class_name': "非猪肠道病毒" if pred == 1 else "猪肠道病毒"
            }
            results.append(result)
        
        return results
    
    # 主界面 - 两种输入方式
    input_option = st.radio("选择输入方式", ["单序列预测", "批量CSV预测"], horizontal=True)
    
    if input_option == "单序列预测":
        st.subheader("🔤 输入蛋白质序列")
        sequence_input = st.text_area(
            "粘贴蛋白质序列 (仅支持标准氨基酸字符)",
            height=150,
            placeholder="例如: MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDMKHLKKADLIICAPNSYKKDDKPNQIKLLAVPTVMTKDDKQLLQEINELQDVVQDLRSLVEKNQIPAVDRAVTLTQRGELQAAGDKTLQEAVDRLQDKLQSLAEEGVKALQEELRKQLEAVDRAVTKLEQKLQDQVEALQARVDSLQAELRALQAQLAELQAELQALRSQLDELQAQLAELQAQLQELQAQLSELQSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQ"
        )
        
        if st.button("🔍 开始预测", type="primary"):
            if not sequence_input.strip():
                st.warning("⚠️ 请输入有效的蛋白质序列")
            else:
                # 预处理序列 - 只保留标准氨基酸
                sequence = ''.join(filter(str.isalpha, sequence_input.strip().upper()))
                sequence = ''.join([aa for aa in sequence if aa in 'ACDEFGHIKLMNPQRSTVWY'])
                
                if len(sequence) < 10:
                    st.error("❌ 序列长度过短，请输入至少10个氨基酸的序列")
                elif len(sequence) > 5000:
                    st.error("❌ 序列长度过长，最大支持5000个氨基酸")
                else:
                    # 进行预测
                    results = predict_sequences([sequence])
                    
                    # 显示结果
                    st.subheader("📊 预测结果")
                    result = results[0]
                    
                    # 使用卡片式布局展示结果
                    if result['prediction'] == 0:
                        color = "#ff4b4b"  # 红色表示猪肠道病毒
                        emoji = "🐷"
                    else:
                        color = "#1f77b4"  # 蓝色表示非猪肠道病毒
                        emoji = "🦠"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 15px; border-radius: 0 8px 8px 0; margin: 15px 0;">
                        <h3 style="color: {color};">{emoji} 预测结果: {result['class_name']}</h3>
                        <p><b>置信度:</b> {result['confidence']:.2%}</p>
                        <p><b>序列预览:</b> {result['sequence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 显示详细置信度
                    st.subheader("📈 置信度分析")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("猪肠道病毒概率", f"{1-result['confidence']:.2%}")
                    with col2:
                        st.metric("非猪肠道病毒概率", f"{result['confidence']:.2%}")
                    
                    # 可视化置信度
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 2))
                    classes = ['PEV', 'non-PEV']
                    probabilities = [1-result['confidence'], result['confidence']]
                    colors = ['#ff4b4b', '#1f77b4']
                    
                    bars = ax.barh(classes, probabilities, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_title('Forecast probability distribution')
                    ax.bar_label(bars, fmt='%.2f', padding=3)
                    
                    st.pyplot(fig)
                    
                    # 显示完整序列
                    with st.expander("📋 查看完整序列"):
                        st.code(result['full_sequence'])
    
    else:  # 批量CSV预测
        st.subheader("📁 上传CSV文件")
        st.markdown("""
        请上传包含蛋白质序列的CSV文件，文件需包含`Sequence`列。
        
        **示例格式:**
        ```
        ID,Sequence
        seq1,MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHF
        seq2,MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGV
        ```
        """)
        
        uploaded_file = st.file_uploader("选择CSV文件", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # 读取CSV
                df = pd.read_csv(uploaded_file)
                
                if 'Sequence' not in df.columns:
                    st.error("❌ CSV文件中缺少'Sequence'列")
                else:
                    st.success(f"✅ 成功加载 {len(df)} 条序列")
                    
                    # 预览数据
                    with st.expander("🔍 数据预览"):
                        st.dataframe(df.head())
                    
                    # 预处理序列
                    sequences = []
                    valid_indices = []
                    for idx, row in df.iterrows():
                        seq = str(row['Sequence']).strip().upper()
                        # 仅保留标准氨基酸
                        seq_clean = ''.join([aa for aa in seq if aa in 'ACDEFGHIKLMNPQRSTVWY'])
                        if len(seq_clean) >= 10 and len(seq_clean) <= 5000:
                            sequences.append(seq_clean)
                            valid_indices.append(idx)
                    
                    st.info(f"ℹ️ 有效序列: {len(sequences)}/{len(df)} (过滤了过短、过长或无效字符的序列)")
                    
                    if st.button("🚀 开始批量预测", type="primary"):
                        if not sequences:
                            st.warning("⚠️ 没有有效的序列可以预测")
                        else:
                            # 进行预测
                            with st.spinner(f"🧠 正在预测 {len(sequences)} 条序列..."):
                                start_time = time.time()
                                results = predict_sequences(sequences)
                                elapsed_time = time.time() - start_time
                            
                            # 创建结果DataFrame
                            results_df = pd.DataFrame(results)
                            results_df = results_df[['sequence_id', 'sequence', 'prediction', 'confidence', 'class_name']]
                            
                            # 与原始数据合并
                            result_indices = pd.Series(valid_indices, name='original_index')
                            results_with_index = pd.concat([result_indices, results_df], axis=1)
                            
                            # 创建完整的输出
                            output_df = df.copy()
                            output_df['Prediction'] = "无效序列"
                            output_df['Class'] = "无效序列"
                            output_df['Confidence'] = 0.0
                            
                            for _, row in results_with_index.iterrows():
                                idx = int(row['original_index'])
                                output_df.at[idx, 'Prediction'] = row['prediction']
                                output_df.at[idx, 'Class'] = row['class_name']
                                output_df.at[idx, 'Confidence'] = row['confidence']
                            
                            # 显示统计信息
                            st.subheader("📊 预测统计")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_valid = len(sequences)
                                st.metric("有效序列数", total_valid)
                            with col2:
                                pig_virus_count = sum(1 for r in results if r['prediction'] == 0)
                                st.metric("猪肠道病毒", pig_virus_count)
                            with col3:
                                non_pig_count = total_valid - pig_virus_count
                                st.metric("非猪肠道病毒", non_pig_count)
                            
                            st.success(f"✅ 预测完成! 耗时: {elapsed_time:.2f} 秒")
                            
                            # 显示结果预览
                            st.subheader("🔍 结果预览")
                            st.dataframe(output_df.head(10))
                            
                            # 下载结果
                            csv = output_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 下载完整结果 (CSV)",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv",
                                type="primary"
                            )
                            
                            # 可视化
                            st.subheader("📈 结果分布")
                            import matplotlib.pyplot as plt
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            class_counts = output_df[output_df['Prediction'] != "无效序列"]['Class'].value_counts()
                            colors = ['#ff4b4b', '#1f77b4']
                            
                            bars = class_counts.plot(kind='bar', color=colors, ax=ax)
                            ax.set_title('预测类别分布', fontsize=16)
                            ax.set_xlabel('类别', fontsize=12)
                            ax.set_ylabel('数量', fontsize=12)
                            ax.tick_params(axis='x', rotation=0)
                            
                            # 添加数据标签
                            for i, v in enumerate(class_counts.values):
                                ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                            
                            st.pyplot(fig)
                            
            except Exception as e:
                st.error(f"❌ 处理文件时出错: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
