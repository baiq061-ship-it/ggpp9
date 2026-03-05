# ==============================
# 必须第一条 Streamlit 命令
# ==============================
import streamlit as st
st.set_page_config(
    page_title="Clinical Prediction System",
    layout="wide",
    page_icon="🏥"
)

# ==============================
# 其它库导入
# ==============================
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 顯示 scikit-learn 版本（方便 debug，上線後可移除）
import sklearn
st.sidebar.write("scikit-learn 版本：", sklearn.__version__)

# ==============================
# 预测因子
# ==============================
FEATURES = [
    "sz1", "sz2", "sz3", "sz4", "sz5",
    "lxsz",
    "cjl0", "cjl1", "cjl2", "cjl3", "cjl4", "cjl5",
    "lb1", "lb2", "lb3", "lb4", "lb5", "lb6",
    "lb7", "lb8", "lb9"
]

# ==============================
# 模型路径
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH_1 = os.path.join(BASE_DIR, "best_model.pkl")
MODEL_PATH_2 = os.path.join(BASE_DIR, "model.pkl")

# ==============================
# 模型加载（缓存）
# ==============================
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH_1):
            return joblib.load(MODEL_PATH_1)
        elif os.path.exists(MODEL_PATH_2):
            return joblib.load(MODEL_PATH_2)
        else:
            raise FileNotFoundError("未找到 best_model.pkl 或 model.pkl")
    except Exception as e:
        st.error(f"模型載入失敗：{str(e)}")
        st.info("常見原因：scikit-learn 版本不兼容。建議在 requirements.txt 指定 scikit-learn==1.2.2")
        raise

model = load_model()

# ==============================
# 页面标题
# ==============================
st.title("🏥 臨床診斷預測系統")

# ==============================
# Pipeline 处理（兼容 scaler 或 pipeline）
# ==============================
def process_input(model, df):
    X = df.copy()
    if hasattr(model, "named_steps"):
        # 如果是 Pipeline，取最後一步前的 transformer 處理輸入
        preprocessor = model[:-1] if len(model.named_steps) > 1 else None
        if preprocessor is not None:
            X = preprocessor.transform(df)
    return X

# ==============================
# 页面 Tabs
# ==============================
tab1, tab2 = st.tabs(["📝 單筆預測", "📂 批量預測"])

# =================================================
# 單筆預測
# =================================================
with tab1:
    st.subheader("單筆病例預測")
    
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                step=0.1,
                format="%.2f"
            )
    
    if st.button("開始預測"):
        with st.spinner("正在預測..."):
            try:
                input_df = pd.DataFrame([input_data])
                X_processed = process_input(model, input_df)
                
                prob = model.predict_proba(X_processed)[0][1]
                
                st.divider()
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.metric("風險機率", f"{prob:.2%}")
                    if prob > 0.5:
                        st.error("高風險")
                    else:
                        st.success("低風險")
                
                with c2:
                    st.info("模型預測完成（SHAP 解釋功能已暫時移除）")
                    
            except Exception as e:
                st.error("預測失敗")
                st.text(str(e))

# =================================================
# 批量預測
# =================================================
with tab2:
    st.subheader("批量預測")
    
    uploaded_file = st.file_uploader(
        "上傳 Excel 或 CSV 檔案（需包含所有特徵欄位）",
        type=["xlsx", "csv"]
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("資料預覽（前 5 行）：")
            st.dataframe(df.head())
            
            missing_cols = [col for col in FEATURES if col not in df.columns]
            if missing_cols:
                st.warning(f"缺少以下必要欄位：{', '.join(missing_cols)}")
            else:
                if st.button("執行批量預測"):
                    with st.spinner("正在批量預測..."):
                        try:
                            X_processed = process_input(model, df[FEATURES])
                            probs = model.predict_proba(X_processed)[:, 1]
                            
                            df["預測風險機率"] = probs
                            df["風險等級"] = np.where(probs > 0.5, "高風險", "低風險")
                            
                            st.success("批量預測完成！")
                            st.dataframe(df)
                            
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="下載結果 CSV",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error("批量預測失敗")
                            st.text(str(e))
        except Exception as e:
            st.error("檔案讀取失敗，請確認格式正確")
            st.text(str(e))