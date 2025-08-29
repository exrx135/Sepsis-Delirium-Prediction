import joblib
import streamlit as st
import pandas as pd
import numpy as np
import json
from catboost import CatBoostClassifier

# 加载模型和阈值配置
try:
    # 使用CatBoost加载.cbm模型
    model = CatBoostClassifier()
    model.load_model('shap_simplified_model.cbm')

    with open('shap_simplified_model_threshold.json', 'r') as f:
        threshold_config = json.load(f)
    optimal_threshold = threshold_config['optimal_threshold']
    feature_names = threshold_config['selected_features']  # 直接从JSON获取特征名称
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 构造 Streamlit 页面
st.set_page_config(page_title="ICU Delirium Prediction Tool", layout="wide")
st.title("🤖 ICU Delirium Prediction Tool (Simplified Model)")
st.title("🤖 ICU 谵妄预测工具（简化模型）")

st.markdown(f"""
**Instructions**:  
Please enter the patient information below. The system will predict the probability of delirium occurring within 7 days after ICU admission based on the SHAP simplified model.  
Number of model features: `{len(feature_names)}`, Optimal threshold: `{optimal_threshold:.4f}`

**说明**:  
请输入以下患者信息，系统将基于SHAP简化模型预测其在ICU后7天内发生谵妄的概率。  
模型特征数：`{len(feature_names)}`，最佳阈值：`{optimal_threshold:.4f}`
""")

# 构建输入表单
user_input = {}
with st.form("input_form"):
    # 使用动态列布局 - 根据特征数量自适应调整
    num_features = len(feature_names)
    num_cols = 3 if num_features > 6 else 2
    cols = st.columns(num_cols)

    for i, feat in enumerate(feature_names):
        col_idx = i % num_cols
        with cols[col_idx]:
            # 为关键临床特征提供默认值和提示
            hint = ""
            if "age" in feat.lower() or "年龄" in feat:
                hint = "(years)"
                default = 60.0
            elif "score" in feat.lower() or "评分" in feat or "分数" in feat:
                hint = "(points)"
                default = 10.0
            else:
                default = 0.0
                hint = "(units)"

            # 创建输入框
            val = st.number_input(
                f"{feat} {hint}",
                value=default,
                format="%.3f",
                key=f"input_{feat}"
            )
            user_input[feat] = val

    submitted = st.form_submit_button("🚀 Predict / 点击预测")

if submitted:
    # 检查输入完整性
    error_msgs = []
    for feat in feature_names:
        if feat not in user_input or pd.isna(user_input[feat]):
            error_msgs.append(f"'{feat}' requires a valid value")
            error_msgs.append(f"'{feat}' 需要提供有效值")

    if error_msgs:
        st.error("**Input Errors / 输入错误:**")
        for msg in error_msgs:
            st.error(msg)
    else:
        try:
            input_df = pd.DataFrame([user_input])
            input_df = input_df[feature_names]  # 确保特征顺序正确

            # 使用模型预测概率
            proba = model.predict_proba(input_df)[:, 1][0]

            # 使用最佳阈值进行二分类预测
            prediction = 1 if proba >= optimal_threshold else 0

            st.divider()
            st.markdown("### 🧠 Prediction Results / 模型预测结果")

            # 用醒目的方式显示概率
            st.markdown(f"""
            <div style="text-align: center;">
                <h2 style="color: {'#FF4B4B' if prediction == 1 else '#0F9D58'};">
                    {proba:.4f}
                </h2>
                <p>Predicted Probability / 预测概率</p>
            </div>
            """, unsafe_allow_html=True)

            if prediction == 1:
                st.error("⚠️ **High Risk: Delirium Likely / 高风险：可能发生谵妄**")
                st.markdown("**Clinical Recommendations / 临床建议:**")
                st.markdown("1. Notify the attending physician immediately\n   立即通知主治医师")
                st.markdown("2. Implement delirium prevention measures\n   实施谵妄预防措施")
                st.markdown("3. Increase monitoring frequency\n   增加监护频率")
            else:
                st.success("✅ **Low Risk: Delirium Unlikely / 低风险：可能不会发生谵妄**")
                st.markdown("**Clinical Recommendations / 临床建议:**")
                st.markdown("1. Maintain routine monitoring\n   维持常规监护")
                st.markdown("2. Observe changes in mental status\n   观察精神状态变化")
                st.markdown("3. Conduct regular risk assessments\n   定期评估风险")

            # 概率风险图
            risk_levels = [0, 0.3, 0.7, 1.0]
            labels = ["Low Risk / 低风险", "Moderate Risk / 中度风险", "High Risk / 高风险"]
            current_level = next(i for i, level in enumerate(risk_levels) if proba < level) - 1

            st.progress(proba, text=f"Risk Level: {labels[current_level]}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error(f"预测过程中发生错误: {str(e)}")
            st.info("Please check the input data format / 请检查输入数据的格式是否正确")

# 添加模型信息侧边栏
with st.sidebar:
    st.header("ℹ️ Model Information / 模型信息")
    st.markdown(f"**Algorithm / 算法**: CatBoost (SHAP Simplified)")
    st.markdown(f"**Number of Features / 特征数**: {len(feature_names)}")
    st.markdown(f"**Optimal Threshold / 最佳阈值**: {optimal_threshold:.4f}")

    st.markdown("### Clinical Features / 临床特征")
    for feat in feature_names:
        st.caption(f"- {feat}")

    st.divider()
    st.markdown("""
    **Development Notes / 开发说明**:  
    This tool was developed based on XXX research data, using SHAP feature selection to identify the most important {len(feature_names)} predictors from the original XX features.

    本工具基于XXX研究数据开发，采用SHAP特征选择方法，从原始XX个特征中筛选出最重要的{len(feature_names)}个预测因子。
    """)

    # 添加研究团队信息
    st.markdown("### Development Team / 开发团队")
    st.caption("- Principal Investigator: Dr. XXX")
    st.caption("- Data Analyst: XXX")
    st.caption("- Clinical Consultant: Prof. XXX")

    # 添加文件下载功能
    st.divider()
    st.markdown("### Download Resources / 资源下载")
    try:
        with open("shap_simplified_model.cbm", "rb") as file:
            st.download_button(
                label="Download Model / 下载模型",
                data=file,
                file_name="delirium_prediction_model.cbm",
                mime="application/octet-stream"
            )
    except:
        st.warning("Model file not available for download / 模型文件不可下载")

    try:
        with open("shap_simplified_model_threshold.json", "rb") as file:
            st.download_button(
                label="Download Config / 下载配置",
                data=file,
                file_name="model_config.json",
                mime="application/json"
            )
    except:
        st.warning("Config file not available for download / 配置文件不可下载")