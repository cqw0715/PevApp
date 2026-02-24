import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from mamba_ssm import Mamba
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心模型架构 (与训练代码完全一致)
# ==========================================
class CNNBranch(nn.Module):
    def __init__(self, input_dim=1280, num_classes=8):
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
        return self.classifier(self.net(x).flatten(1))

class TransformerBranch(nn.Module):
    def __init__(self, input_dim=1280, d_model=256, nhead=8, num_classes=8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        return self.classifier(self.transformer(x).squeeze(1))

class MambaBranch(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        self.preprocess = nn.Linear(input_dim, 256)
        self.mamba_blocks = nn.ModuleList([Mamba(d_model=256, d_state=16, d_conv=4, expand=2) for _ in range(5)])
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.preprocess(x).unsqueeze(1)
        for block in self.mamba_blocks:
            x = x + block(x)
        return self.classifier(self.norm(x).squeeze(1))

class MutualLearningModel(nn.Module):
    def __init__(self, input_dim, num_classes=8, embed_dim=128):
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
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim, 8, dropout=0.2, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim*4),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim*4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            }) for _ in range(2)
        ])
        self.gate = nn.Sequential(
            nn.Linear(embed_dim*3 + num_classes*3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.8)))
        self.refine = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LayerNorm(num_classes),
            nn.GELU()
        )
    
    def forward(self, x):
        o1, o2, o3 = self.cnn(x), self.trans(x), self.mamba(x)
        branches = torch.stack([o1, o2, o3], dim=1)
        
        x_f = self.feature_proj(self.logits_norm(branches))
        for b in self.blocks:
            attn_out, _ = b['attn'](x_f, x_f, x_f)
            x_f = b['norm1'](x_f + attn_out)
            x_f = b['norm2'](x_f + b['ffn'](x_f))
        
        gate_input = torch.cat([x_f.flatten(1), branches.flatten(1)], dim=1)
        temp = F.softplus(self.log_temp) + 1e-4
        weights = F.softmax(self.gate(gate_input) / temp, dim=1).unsqueeze(-1)
        o_fused = (branches * weights).sum(dim=1)
        return o1, o2, o3, o_fused + self.refine(o_fused)

# ==========================================
# 2. ESM 特征提取器
# ==========================================
class ESMFeatureExtractor:
    def __init__(self):
        self.gpu_model = None
        self.gpu_batch_converter = None
        self.cpu_model = None
        self.cpu_batch_converter = None
        self.device = None
        self._initialize_models()
        
    def _initialize_models(self):
        try:
            if torch.cuda.is_available():
                print("🚀 尝试加载GPU模型（ESM-2 650M）...")
                self.gpu_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.gpu_device = torch.device('cuda')
                self.gpu_model = self.gpu_model.to(self.gpu_device)
                self.gpu_batch_converter = alphabet.get_batch_converter()
                self.device = self.gpu_device
                print("✅ GPU模型加载成功")
        except Exception as e:
            print(f"❌ GPU模型加载失败: {e}")
        
        try:
            print("🖥️ 加载CPU模型作为备用...")
            self.cpu_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.cpu_device = torch.device('cpu')
            self.cpu_model = self.cpu_model.to(self.cpu_device)
            self.cpu_batch_converter = alphabet.get_batch_converter()
            if self.device is None: self.device = self.cpu_device
            print("✅ CPU模型加载成功")
        except Exception as e:
            print(f"❌ CPU模型加载失败: {e}")
            raise

    def _extract_batch_features(self, batch_data, use_gpu=True):
        try:
            model = self.gpu_model if use_gpu and self.gpu_model else self.cpu_model
            batch_converter = self.gpu_batch_converter if use_gpu and self.gpu_model else self.cpu_batch_converter
            device = self.gpu_device if use_gpu and self.gpu_model else self.cpu_device
                
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                seq_lengths = (batch_tokens != model.alphabet.padding_idx).sum(1)
                batch_features = [token_representations[i, :seq_lengths[i]].mean(0).cpu().numpy() 
                                 for i in range(token_representations.size(0))]
            
            del batch_tokens, results
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return batch_features
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and use_gpu:
                return self._extract_batch_features(batch_data, use_gpu=False)
            raise

    def extract_features(self, sequences, cache_path=None, batch_size=1):
        if cache_path and os.path.exists(cache_path):
            print(f"📂 从缓存加载特征: {cache_path}")
            with open(cache_path, 'rb') as f: return pickle.load(f)
        
        features = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
            features.extend(self._extract_batch_features(batch_data))
            if (i // batch_size) % 10 == 0: print(f"📊 进度: {min(i+batch_size, len(sequences))}/{len(sequences)}")
            
        features_array = np.array(features)
        if cache_path:
            with open(cache_path, 'wb') as f: pickle.dump(features_array, f)
        return features_array

# ==========================================
# 3. CSV处理专用函数
# ==========================================
def validate_sequence(seq):
    """验证蛋白质序列"""
    seq = seq.strip().upper()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    invalid_chars = [c for c in seq if c not in valid_aa]
    
    if invalid_chars:
        return False, f"无效字符: {', '.join(set(invalid_chars))}"
    if len(seq) < 10:
        return False, "序列太短 (至少需要10个氨基酸)"
    if len(seq) > 10000:
        return False, "序列太长 (最多10000个氨基酸)"
    
    return True, ""

def validate_csv_sequences(sequences, seq_names):
    """验证CSV中的序列，返回有效序列索引和错误信息"""
    valid_indices = []
    errors = []
    
    for i, seq in enumerate(sequences):
        is_valid, message = validate_sequence(seq)
        if is_valid:
            valid_indices.append(i)
        else:
            errors.append((seq_names[i], message))
    
    return valid_indices, errors

def parse_csv_sequences(uploaded_file):
    """
    解析上传的CSV文件，智能识别序列列和名称列
    返回: (序列名称列表, 序列列表, 原始DataFrame, 序列列名, 名称列名)
    """
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ 成功读取CSV文件，共 {len(df)} 行 {len(df.columns)} 列")
        
        # 查找序列列（不区分大小写）
        seq_col = None
        name_col = None
        possible_seq_cols = ['sequence', 'seq', 'protein_sequence', 'aa_sequence', 'peptide', 'protein']
        possible_name_cols = ['name', 'id', 'protein_id', 'identifier', 'accession', 'entry']
        
        # 查找序列列
        for col in df.columns:
            if col.lower() in possible_seq_cols:
                seq_col = col
                break
        
        # 如果未找到，尝试查找包含"seq"的列
        if seq_col is None:
            for col in df.columns:
                if 'seq' in col.lower() or 'sequence' in col.lower():
                    seq_col = col
                    break
        
        # 查找名称列
        for col in df.columns:
            if col.lower() in possible_name_cols:
                name_col = col
                break
        
        # 如果仍未找到序列列，报错
        if seq_col is None:
            st.error("❌ 未检测到序列列。请确保CSV包含以下列名之一：'Sequence', 'Seq', 'Protein_Sequence'等")
            st.info("💡 提示：列名不区分大小写，且需包含蛋白质氨基酸序列")
            return None, None, None, None, None
        
        # 提取序列（清理空格和NaN）
        sequences = []
        for idx, seq in enumerate(df[seq_col]):
            if pd.isna(seq) or str(seq).strip() == "":
                st.warning(f"⚠️ 第 {idx+1} 行序列为空，将跳过")
                sequences.append(None)
            else:
                sequences.append(str(seq).strip().upper())
        
        # 生成名称列表
        if name_col is not None:
            seq_names = []
            for idx, name in enumerate(df[name_col]):
                if pd.isna(name) or str(name).strip() == "":
                    seq_names.append(f"Seq_{idx+1}")
                else:
                    seq_names.append(str(name).strip())
        else:
            seq_names = [f"Seq_{i+1}" for i in range(len(sequences))]
        
        # 过滤空序列
        valid_indices = [i for i, seq in enumerate(sequences) if seq is not None and len(seq.strip()) > 0]
        filtered_names = [seq_names[i] for i in valid_indices]
        filtered_seqs = [sequences[i] for i in valid_indices]
        
        name_display = "自动编号" if name_col is None else f'"{name_col}"'
        st.info(f"🔍 检测到序列列: '{seq_col}' | 名称列: {name_display}")
        st.info(f"✅ 有效序列数量: {len(filtered_seqs)} / {len(sequences)}")
        
        return filtered_names, filtered_seqs, df, seq_col, name_col
    
    except Exception as e:
        st.error(f"❌ 解析CSV文件失败: {str(e)}")
        st.info("💡 请确保文件是有效的CSV格式，且包含蛋白质序列列")
        return None, None, None, None, None

# ==========================================
# 4. 模型加载
# ==========================================
@st.cache_resource
def load_model_and_scaler():
    """加载模型和标准化器，使用缓存提高性能"""
    import numpy as np
    import numpy.core.multiarray
    import sklearn.preprocessing._data
    
    safe_globals = [
        np.core.multiarray.scalar,
        np.dtype,
        np.ndarray,
        StandardScaler,
        sklearn.preprocessing._data.StandardScaler
    ]
    
    for obj in safe_globals:
        try:
            torch.serialization.add_safe_globals([obj])
        except Exception as e:
            st.warning(f"无法添加安全全局变量: {str(e)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"使用设备: {device}")
    
    model_path = "best_multiclass_model.pth"
    if not os.path.exists(model_path):
        st.error(f"模型文件 {model_path} 未找到！请确保文件在当前目录中。")
        st.stop()
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        st.success("✅ 模型安全加载成功 (使用weights_only=True)")
    except Exception as e:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e2:
            st.error(f"❌ 两种加载方式都失败: {str(e2)}")
            st.stop()
    
    virus_map = checkpoint.get('virus_map', {
        0: "Adenovirus",
        1: "Herpesvirus", 
        2: "Orthomyxovirus",
        3: "Papillomavirus",
        4: "Picornavirus",
        5: "Polyomavirus",
        6: "Rotavirus",
        7: "Coronavirus"
    })
    st.info(f"病毒类别映射: {', '.join(virus_map.values())}")
    
    model = MutualLearningModel(input_dim=1280, num_classes=8).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    return model, scaler, virus_map, device

# ==========================================
# 5. 预测和可视化函数
# ==========================================
def predict(model, scaler, sequences, device, virus_map):
    """进行预测"""
    extractor = ESMFeatureExtractor()
    st.info("🧬 正在提取ESM-2特征，请稍候...")
    features = extractor.extract_features(sequences)
    
    st.info("⚖️ 标准化特征...")
    scaled_features = scaler.transform(features)
    
    st.info("🧠 进行预测...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(len(scaled_features)):
            x = torch.FloatTensor(scaled_features[i:i+1]).to(device)
            _, _, _, fused_output = model(x)
            probs = F.softmax(fused_output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            results.append({
                'probabilities': probs,
                'predicted_class': virus_map[pred_idx],
                'confidence': probs[pred_idx]
            })
    
    return results

def create_probability_chart(probs, virus_map, title="类别概率分布"):
    """使用纯matplotlib创建概率分布图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    viruses = [virus_map[i] for i in range(len(probs))]
    colors = ['red' if i == np.argmax(probs) else 'steelblue' for i in range(len(probs))]
    
    bars = ax.bar(viruses, probs, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('预测概率', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.02,
            f'{prob:.2f}',
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig

# ==========================================
# 6. Streamlit 应用主函数
# ==========================================
def main():
    st.set_page_config(
        page_title="病毒蛋白分类器",
        page_icon="🦠",
        layout="wide"
    )
    
    st.title("🦠 病毒蛋白多分类预测系统")
    st.markdown("""
    该系统使用深度学习模型对病毒蛋白序列进行分类，支持8种病毒家族的识别。
    请上传包含蛋白质序列的CSV文件或直接输入单条序列进行预测。
    """)
    
    with st.spinner("⏳ 加载模型和相关组件..."):
        try:
            model, scaler, virus_map, device = load_model_and_scaler()
        except Exception as e:
            st.error(f"加载模型时发生严重错误: {str(e)}")
            st.stop()
    
    st.success("✅ 模型加载成功！")
    
    tab1, tab2, tab3 = st.tabs(["🔬 单序列预测", "📁 批量预测 (CSV)", "ℹ️ 关于模型"])
    
    with tab1:
        st.header("单序列预测")
        sequence_input = st.text_area(
            "输入蛋白质序列 (氨基酸序列)",
            height=150,
            placeholder="例如: MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHF..."
        )
        
        if st.button("🚀 预测", type="primary", use_container_width=True):
            if not sequence_input.strip():
                st.warning("⚠️ 请输入有效的蛋白质序列")
            else:
                is_valid, message = validate_sequence(sequence_input)
                if not is_valid:
                    st.error(f"❌ 序列无效: {message}")
                else:
                    with st.spinner("⏳ 处理中..."):
                        start_time = time.time()
                        results = predict(model, scaler, [sequence_input], device, virus_map)
                        elapsed_time = time.time() - start_time
                    
                    res = results[0]
                    st.subheader("🎯 预测结果")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            "预测病毒家族", 
                            res['predicted_class'],
                            delta=f"{res['confidence']:.1%} 置信度"
                        )
                        st.caption(f"⏱️ 处理时间: {elapsed_time:.2f} 秒")
                    
                    with col2:
                        fig = create_probability_chart(
                            res['probabilities'], 
                            virus_map,
                            f"序列预测概率分布 (置信度: {res['confidence']:.1%})"
                        )
                        st.pyplot(fig)
                    
                    st.subheader("📊 详细概率")
                    prob_df = pd.DataFrame({
                        '病毒家族': [virus_map[i] for i in range(8)],
                        '概率': res['probabilities']  # 保留为 float
                    }).sort_values('概率', ascending=False).reset_index(drop=True)
                    # 安全格式化：仅对数值列应用格式
                    st.dataframe(
                        prob_df.style.format({'概率': '{:.4f}'}),
                        use_container_width=True
                    )
    
    with tab2:
        st.header("批量预测 (CSV格式)")
        st.markdown("""
        **上传包含蛋白质序列的CSV文件**  
        ✅ 必需列: 包含氨基酸序列的列（列名如 `Sequence`, `Protein_Sequence`, `seq` 等）  
        ✅ 可选列: 序列标识列（列名如 `Name`, `ID`, `Accession` 等）  
        
        **CSV示例:**
        ```csv
        Name,Sequence
        Spike_1,MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHF...
        Capsid_2,MKLKKKVVVAVVAVVAGVFVAAVAGVFAAAGVFAAGVFAAGVFAAGVFAAGVFAAGVFAAGVFAAGV...
        ```
        """)
        
        uploaded_file = st.file_uploader(
            "📤 上传CSV文件 (包含Sequence列)", 
            type=["csv"],
            help="CSV文件必须包含蛋白质序列列，列名可为Sequence/Seq/Protein_Sequence等"
        )
        
        if uploaded_file is not None:
            seq_names, sequences, raw_df, seq_col, name_col = parse_csv_sequences(uploaded_file)
            
            if sequences is None or len(sequences) == 0:
                st.stop()
            
            with st.expander("🔍 CSV数据预览 (前10行)"):
                preview_df = raw_df.head(10).copy()
                st.dataframe(preview_df, use_container_width=True)
                name_info = "未检测到名称列，将使用自动编号" if name_col is None else f"名称列: {name_col}"
                st.caption(f"检测到序列列: '{seq_col}' | {name_info}")
            
            st.info(f"📊 共检测到 {len(sequences)} 个有效序列")
            
            if st.button("🚀 开始批量预测", type="primary", use_container_width=True):
                valid_indices, errors = validate_csv_sequences(sequences, seq_names)
                
                if errors:
                    st.error(f"❌ 发现 {len(errors)} 个无效序列:")
                    for name, msg in errors[:10]:
                        st.write(f"- **{name}**: {msg}")
                    if len(errors) > 10:
                        st.write(f"... 还有 {len(errors)-10} 个错误未显示")
                    st.stop()
                
                if len(valid_indices) > 50:
                    st.warning(f"⚠️ 您上传了 {len(valid_indices)} 个序列，处理可能需要较长时间")
                
                with st.spinner(f"⏳ 正在预测 {len(valid_indices)} 个序列..."):
                    start_time = time.time()
                    valid_seqs = [sequences[i] for i in valid_indices]
                    valid_names = [seq_names[i] for i in valid_indices]
                    results = predict(model, scaler, valid_seqs, device, virus_map)
                    total_time = time.time() - start_time
                
                # ====== 修复核心：保留数值类型，不在构建时转字符串 ======
                results_data = []
                for i, (name, res) in enumerate(zip(valid_names, results)):
                    row = {
                        '序列名称': name,
                        '预测病毒': res['predicted_class'],
                        '置信度': res['confidence']  # 保留为 float
                    }
                    # 添加所有病毒家族概率（保留为 float）
                    for j in range(8):
                        row[virus_map[j]] = res['probabilities'][j]  # 关键修复：不转字符串
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                
                st.subheader("📈 预测结果汇总")
                st.caption(f"⏱️ 总耗时: {total_time:.2f} 秒 | 平均: {total_time/len(valid_indices):.2f} 秒/序列")
                
                # ====== 安全格式化：显式构建格式化字典 ======
                format_dict = {'置信度': '{:.2%}'}  # 置信度显示为百分比
                # 为所有病毒家族列添加格式（排除非数值列）
                for col in results_df.columns:
                    if col not in ['序列名称', '预测病毒', '置信度']:
                        format_dict[col] = '{:.4f}'
                
                # 应用格式化（添加 na_rep 处理潜在缺失值）
                styled_df = results_df.style.format(format_dict, na_rep='N/A')
                st.dataframe(styled_df, use_container_width=True)
                
                st.subheader("📊 可视化选项")
                col1, col2 = st.columns(2)
                with col1:
                    show_chart = st.checkbox("显示所有序列预测概览", value=True)
                with col2:
                    if len(valid_names) > 1:
                        show_details = st.checkbox("查看单个序列详细分布")
                
                if show_chart and len(valid_indices) <= 20:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    x = np.arange(len(virus_map))
                    width = 0.8 / len(valid_names)
                    
                    for i, (name, res) in enumerate(zip(valid_names, results)):
                        ax.bar(x + i*width, res['probabilities'], width, label=name)
                    
                    ax.set_xlabel('病毒家族')
                    ax.set_ylabel('预测概率')
                    ax.set_title('所有序列预测概率对比')
                    ax.set_xticks(x + width * (len(valid_names)-1)/2)
                    ax.set_xticklabels([virus_map[i] for i in range(8)], rotation=30, ha='right')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.set_ylim(0, 1.05)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if show_details and len(valid_names) > 1:
                    selected_seq = st.selectbox(
                        "选择要查看详细分布的序列",
                        options=valid_names,
                        key="seq_selector_csv"
                    )
                    idx = valid_names.index(selected_seq)
                    fig = create_probability_chart(
                        results[idx]['probabilities'],
                        virus_map,
                        f"{selected_seq} 的预测概率分布"
                    )
                    st.pyplot(fig)
                
                # 下载保留原始数值（小数形式，便于后续分析）
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载预测结果 (CSV)",
                    data=csv,
                    file_name="virus_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with tab3:
        st.header("ℹ️ 关于模型")
        st.markdown("""
        ### 🧠 模型架构
        - **三分支融合架构**: CNN + Transformer + Mamba
        - **自适应门控融合**: 动态加权整合三个分支的预测
        - **输入特征**: ESM-2 (650M) 提取的1280维蛋白质表示
        
        ### 🦠 支持的病毒家族 (8类)
        | 编号 | 病毒家族 | 常见代表 |
        |------|----------|----------|
        | 0 | Adenovirus | 腺病毒 |
        | 1 | Herpesvirus | 疱疹病毒 |
        | 2 | Orthomyxovirus | 流感病毒 |
        | 3 | Papillomavirus | 人乳头瘤病毒 |
        | 4 | Picornavirus | 肠道病毒 |
        | 5 | Polyomavirus | 多瘤病毒 |
        | 6 | Rotavirus | 轮状病毒 |
        | 7 | Coronavirus | 冠状病毒 |
        
        ### 📊 CSV上传说明
        - **必需列**: 包含氨基酸序列的列（自动识别常见列名）
        - **智能识别**: 支持多种列名变体（不区分大小写）
        - **错误处理**: 自动跳过空序列，详细报告无效序列
        - **名称处理**: 优先使用ID列，无ID时自动生成序列名称
        
        ### 🔒 安全说明
        - 模型加载使用 PyTorch `weights_only=True` 安全模式
        - 通过 `torch.serialization.add_safe_globals()` 安全加载 StandardScaler
        - 所有预测在本地完成，数据不会上传到外部服务器
        
        ### 📦 依赖要求
        ```bash
        pip install streamlit torch esm mamba-ssm pandas numpy scikit-learn matplotlib
        ```
        """)

if __name__ == "__main__":
    main()
