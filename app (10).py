import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

st.set_page_config(page_title="AI Network Intrusion Detection System", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_nids_components():
    try:
        model = joblib.load('nids_model.pkl')
    except:
        with open('nids_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
    with open('nids_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_nids_components()
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

st.title("🛡️ AI-Powered Network Intrusion Detection System (NIDS)")
st.markdown("Analyze network traffic logs to detect Cyber Threats, Botnets, and Malicious Traffic using ML.")

st.write("---")

tab1, tab2 = st.tabs(["📁 Batch CSV Analysis", "⚙️ Manual Feature Testing"])

with tab1:
    st.header("Upload Traffic Logs (CSV)")
    uploaded_file = st.file_uploader("Upload CSV containing network packets", type=["csv"])

    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            st.dataframe(df_in.head(5), use_container_width=True)

            df_in.columns = df_in.columns.str.strip()
            df_in.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_in.dropna(inplace=True)

            for col in feature_names:
                if col not in df_in.columns:
                    df_in[col] = 0

            X_in = df_in[feature_names]

            if st.button("Analyze Logs", type="primary"):
                scaled_x = scaler.transform(X_in)
                preds = model.predict(scaled_x)

                df_in['Prediction'] = np.where(preds == 1, '🔴 Attack Detected', '🟢 Normal Traffic')

                total = len(preds)
                attacks = np.sum(preds == 1)
                normals = np.sum(preds == 0)

                st.subheader("📊 Analysis Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Packets", f"{total:,}")
                c2.metric("Normal Packets", f"{normals:,}")
                c3.metric("Attacks Detected", f"{attacks:,}")

                if attacks > 0:
                    st.error(f"⚠️ **ALERT:** {attacks} Malicious Traffic packets detected!")
                else:
                    st.success("✅ **SAFE:** Traffic baseline is normal.")

                st.dataframe(df_in, use_container_width=True)
        except Exception as e:
            st.error(f"Processing Error: {e}")

with tab2:
    st.header("Manual Packet Input Test")
    user_inputs = {}
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flow Metrics")
        for i, feat in enumerate(feature_names[:len(feature_names)//2]):
            user_inputs[feat] = st.number_input(f"{feat}", value=0.0, key=f"f_{i}")

    with col2:
        st.subheader("Packet Metrics")
        for i, feat in enumerate(feature_names[len(feature_names)//2:]):
            user_inputs[feat] = st.number_input(f"{feat}", value=0.0, key=f"f_{i+50}")

    if st.button("Predict Packet Threat", type="primary"):
        single_df = pd.DataFrame([user_inputs])[feature_names]
        scaled_single = scaler.transform(single_df)
        pred = model.predict(scaled_single)[0]

        if pred == 1:
            st.error("🔴 **ALERT:** Malicious Attack Traffic Detected!")
        else:
            st.success("🟢 **CLEAN:** Normal Network Packet Traffic.")
