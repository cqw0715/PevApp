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
# 模型架构定义（修改为480维输入，适配ESM-2 35M）
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
    def __init__(self, input_dim=480, num_classes=2):  # ✅ 修改：input_dim=480
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
    def __init__(self, input_dim=480, d_model=256, nhead=8, num_classes=2):  # ✅ 修改：input_dim=480
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
    def __init__(self, input_dim=480, num_classes=2, embed_dim=128):  # ✅ 修改：input_dim=480
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
# 特征提取类（使用ESM-2 35M，输出480维特征）
# ==========================================
class ESMFeatureExtractor:
    """改进的ESM特征提取器，使用ESM-2 35M模型，支持GPU异常后切换到CPU继续提取"""
    def __init__(self):
        self.gpu_model = None
        self.gpu_batch_converter = None
        self.cpu_model = None
        self.cpu_batch_converter = None
        self.device = None
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化GPU和CPU模型（使用35M版本）"""
        try:
            # 先尝试加载GPU模型
            if torch.cuda.is_available():
                print("🚀 尝试加载GPU模型（ESM-2 35M）...")
                # 使用35M参数模型 (12 layers, 480维输出)
                self.gpu_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.gpu_device = torch.device('cuda')
                self.gpu_model = self.gpu_model.to(self.gpu_device)
                self.gpu_batch_converter = alphabet.get_batch_converter()
                self.device = self.gpu_device
                print("✅ GPU模型（ESM-2 35M）加载成功")
            else:
                print("ℹ️ CUDA不可用，直接使用CPU")
        except Exception as e:
            print(f"❌ GPU模型加载失败: {e}")
        
        # 总是加载CPU模型作为备用
        try:
            print("🖥️ 加载CPU模型（ESM-2 35M）作为备用...")
            self.cpu_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.cpu_device = torch.device('cpu')
            self.cpu_model = self.cpu_model.to(self.cpu_device)
            self.cpu_batch_converter = alphabet.get_batch_converter()
            if self.device is None:
                self.device = self.cpu_device
            print("✅ CPU模型（ESM-2 35M）加载成功")
        except Exception as e:
            print(f"❌ CPU模型加载失败: {e}")
            raise

    def _extract_batch_features(self, batch_data, use_gpu=True):
        """提取单个批次的特征（输出480维）"""
        try:
            if use_gpu and self.gpu_model is not None:
                model = self.gpu_model
                batch_converter = self.gpu_batch_converter
                device = self.gpu_device
            else:
                model = self.cpu_model
                batch_converter = self.cpu_batch_converter
                device = self.cpu_device
                
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                # 使用第12层（35M模型的最后一层）进行特征提取，输出480维
                results = model(batch_tokens, repr_layers=[12], return_contacts=False)
                token_representations = results["representations"][12]
                
                # 平均池化（忽略填充token）
                seq_lengths = (batch_tokens != model.alphabet.padding_idx).sum(1)
                batch_features = []
                for seq_idx in range(token_representations.size(0)):
                    seq_len = seq_lengths[seq_idx].item()
                    seq_rep = token_representations[seq_idx, :seq_len]
                    batch_features.append(seq_rep.mean(0).cpu().numpy())
            
            # 清理内存
            del batch_tokens, results, token_representations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return batch_features
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"💥 GPU内存不足，尝试使用CPU...")
                if use_gpu:
                    return self._extract_batch_features(batch_data, use_gpu=False)
                else:
                    raise
            else:
                raise

    def extract_features(self, sequences, cache_path=None, batch_size=1):
        """智能特征提取：支持断点续传和故障转移（输出480维特征）"""
        progress_file = None
        if cache_path:
            progress_file = cache_path.replace('.pkl', '_progress.pkl')
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
        
        if cache_path and os.path.exists(cache_path):
            print(f"📂 从缓存加载完整特征: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        start_idx = 0
        features = []
        if progress_file and os.path.exists(progress_file):
            print(f"🔄 检测到进度文件，尝试恢复特征提取...")
            try:
                with open(progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                features = progress_data['features']
                start_idx = progress_data['last_index'] + 1
                print(f"✅ 从第 {start_idx} 个序列恢复，已完成 {len(features)} 个特征")
            except Exception as e:
                print(f"❌ 加载进度文件失败: {e}")
                start_idx = 0
                features = []
        
        if start_idx >= len(sequences):
            print("✅ 所有特征已提取完成")
            return np.array(features)
        
        print(f"🔧 开始特征提取（使用ESM-2 35M模型，输出480维特征）... (从 {start_idx}/{len(sequences)})")
        
        i = start_idx
        use_gpu = (self.gpu_model is not None)
        
        while i < len(sequences):
            batch_end = min(i + batch_size, len(sequences))
            batch = sequences[i:batch_end]
            batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
            
            try:
                batch_features = self._extract_batch_features(batch_data, use_gpu=use_gpu)
                features.extend(batch_features)
                
                if progress_file:
                    with open(progress_file, 'wb') as f:
                        pickle.dump({'features': features, 'last_index': batch_end - 1}, f)
                
                if (i // batch_size) % 10 == 0:
                    device_type = "GPU" if use_gpu and self.gpu_model is not None else "CPU"
                    print(f"📊 {device_type}模式 - 已完成 {batch_end}/{len(sequences)}")
                
                i = batch_end
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and use_gpu:
                    print("💥 GPU内存不足，切换到CPU模式继续提取...")
                    use_gpu = False
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"❌ 处理批次 {i} 时出错: {e}")
                    i = batch_end
                    continue
            except Exception as e:
                print(f"❌ 处理批次 {i} 时未知错误: {e}")
                i = batch_end
                continue
        
        features_array = np.array(features)
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(features_array, f)
            if progress_file and os.path.exists(progress_file):
                os.remove(progress_file)
        
        print(f"✅ 特征提取完成！特征维度: {features_array.shape} (应为[样本数, 480])")
        return features_array

# ==========================================
# 缓存函数 - 正确使用st.cache_resource
# ==========================================
@st.cache_resource
def get_feature_extractor():
    """获取特征提取器的缓存实例（ESM-2 35M）"""
    return ESMFeatureExtractor()

@st.cache_resource
def load_model_and_scaler():
    """加载预训练模型和标准化器（适配480维输入）"""
    with st.spinner("🔄 正在加载预训练模型（480维输入）..."):
        # 检查模型文件是否存在
        model_path = "best_mutual_learning_model.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件未找到: {model_path}")
            st.info("请确保模型文件与应用在同一目录下，且是使用480维特征训练的模型")
            return None, None, None
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        
        # 初始化480维输入的模型
        model = MutualLearningModel(input_dim=480, num_classes=2).to(device)  # ✅ 关键修改：input_dim=480
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            st.error(f"❌ 模型权重加载失败: {e}")
            st.error("请确认模型文件是使用480维输入训练的！")
            return None, None, None
        model.eval()
        
        # 获取标准化器（应为480维）
        scaler = checkpoint['scaler']
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != 480:
            st.warning(f"⚠️ 警告：标准化器期望 {scaler.n_features_in_} 维输入，但当前使用480维特征！")
            st.warning("请确保使用与训练时相同维度的特征和标准化器")
        
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
            <li><b>类别0</b>: 猪肠道病毒 (PEV)</li>
            <li><b>类别1</b>: 非猪肠道病毒 (non-PEV)</li>
        </ul>
        <p>模型基于<strong>ESM-2 35M</strong>特征提取器（输出480维特征）和多分支融合架构，提供高精度的预测结果。</p>
        <p style="color: #e74c3c; font-weight: bold;">⚠️ 重要：请确保模型文件是使用480维特征训练的！</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        st.markdown("### 模型信息")
        st.info("深度学习融合模型\n(ESM-2 35M + CNN + Transformer + Mamba)\n<strong>输入维度: 480</strong>")
        
        st.markdown("### 使用说明")
        st.markdown("""
        1. **单序列预测**: 在输入框中粘贴蛋白质序列
        2. **批量预测**: 上传包含序列的CSV文件
        3. 查看预测结果及置信度
        """)
        
        st.markdown("### 注意事项")
        st.warning("""
        - 仅支持标准氨基酸字符 (ACDEFGHIKLMNPQRSTVWY)
        - 序列长度建议在50-2000个氨基酸之间
        - GPU加速可显著提升处理速度
        - <strong>必须使用480维训练的模型文件</strong>
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
        
        # 提取特征（480维）
        with st.spinner(f"🧬 正在提取 {len(sequences)} 条序列的特征（ESM-2 35M，480维）..."):
            features = feature_extractor.extract_features(sequences)
        
        # 验证特征维度
        if features.shape[1] != 480:
            st.error(f"❌ 特征维度错误！期望480维，但得到{features.shape[1]}维")
            st.stop()
        
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
            "粘贴蛋白质序列 (仅支持标准氨基酸字符 ACDEFGHIKLMNPQRSTVWY)",
            height=150,
            placeholder="例如: MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDMKHLKKADLIICAPNSYKKDDKPNQIKLLAVPTVMTKDDKQLLQEINELQDVVQDLRSLVEKNQIPAVDRAVTLTQRGELQAAGDKTLQEAVDRLQDKLQSLAEEGVKALQEELRKQLEAVDRAVTKLEQKLQDQVEALQARVDSLQAELRALQAQLAELQAELQALRSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQALQSELQAQLSQLDELQAQLAELQAQLQ"
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
                        <p><small>💡 系统使用ESM-2 35M模型提取480维特征进行预测</small></p>
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
                    ax.set_title('预测概率分布')
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
                            with st.spinner(f"🧠 正在预测 {len(sequences)} 条序列（使用480维特征）..."):
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
                            
                            st.success(f"✅ 预测完成! 耗时: {elapsed_time:.2f} 秒 | 特征维度: 480")
                            
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
                            valid_results = output_df[output_df['Prediction'] != "无效序列"]
                            class_counts = valid_results['Class'].value_counts()
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
