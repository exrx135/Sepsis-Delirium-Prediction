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

# 标题部分 - 使用更大的字体和加粗
st.markdown("<h1 style='text-align: center;'>Delirium Prediction Tool for Sepsis Patients in ICU</h1>",
            unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>ICU脓毒症患者谵妄预测工具</h2>", unsafe_allow_html=True)

# 说明部分 - 使用加粗和稍大字体
st.markdown("""
<div style="font-size: 1.1em;">
    <p><strong>Instructions:</strong><br>
    Please enter the patient information below. The system will predict the probability of delirium occurring within 7 days after ICU admission for sepsis patients based on the externally validated SHAP interpretable model.<br>
    All measured variables should use average values from the first 24 hours after ICU admission (except GCS which uses the lowest value within the first 24 hours).</p>

    说明:
    请输入以下的患者信息，系统将基于经外部验证的SHAP可解释性模型预测ICU脓毒症患者在ICU入科7天内发生谵妄的概率。
    以下所有需测量的变量均需使用入ICU后第1天内的平均值数据（GCS为取入ICU1天内最低值）。
</div>
""", unsafe_allow_html=True)

# 定义特征分组和显示名称
feature_groups = {
    "Basic Information / 基础情况": [
        ("admission_age", "Age / 年龄 (years)"),
        ("hypertension", "Hypertension / 高血压")
    ],
    "Bedside Signs / 床旁体征": [
        ("sbp", "Systolic Blood Pressure (SBP) / 收缩压 (mmHg)"),
        ("spo2", "Oxygen Saturation (SpO₂) / 血氧饱和度 (%)"),
        ("temperature", "Temperature / 体温 (℃)"),
        ("urineoutput_24h", "24-hour Urine Output / 24小时内尿量 (ml)")
    ],
    "Laboratory Tests / 实验室检查": [
        ("platelet", "Platelet Count / 血小板计数 (K/μL)"),
        ("creatinine", "Serum Creatinine / 血清肌酐 (mg/dL, 1mg/dL=88.4μmol/L)"),
        ("potassium", "Potassium Concentration / 钾离子浓度 (mmol/L)"),
        ("hemoglobin", "Hemoglobin / 血红蛋白 (g/dL)")
    ],
    "Clinical Scores / 临床评分": [
        ("charlson_comorbidity_index", "Charlson Comorbidity Index / Charlson共病指数"),
        ("gcs_min", "Glasgow Coma Scale (GCS) Min / 最低格拉斯哥昏迷评分"),
        ("apsiii", "Acute Physiology Score III (APS III) / 急性生理评分III"),
        ("oasis", "Oxford Acute Disease Severity Score (OASIS) / 牛津急性疾病严重程度评分")
    ]
}

# 构建输入表单
user_input = {}
with st.form("input_form"):
    # 为每个特征组创建部分
    for group_name, features in feature_groups.items():
        st.subheader(group_name)
        cols = st.columns(2)  # 每组使用两列布局

        for i, (raw_name, display_name) in enumerate(features):
            col_idx = i % 2
            with cols[col_idx]:
                # 设置默认值和约束
                default = 0.0
                min_value = 0.0
                max_value = None
                step = 0.1

                # 特殊处理某些特征
                if "hypertension" in raw_name.lower():
                    # 高血压 - 单选按钮
                    hypertension_val = st.radio(
                        display_name,
                        options=[0, 1],
                        index=0,
                        horizontal=True,
                        key=f"input_{raw_name}"
                    )
                    user_input[raw_name] = hypertension_val
                elif "gcs_min" in raw_name.lower():
                    # GCS评分 - 滑块
                    gcs_val = st.slider(
                        display_name,
                        min_value=3,
                        max_value=15,
                        value=15,
                        step=1,
                        key=f"input_{raw_name}"
                    )
                    user_input[raw_name] = gcs_val
                elif "charlson_comorbidity_index" in raw_name.lower():
                    # Charlson共病指数 - 整数输入
                    charlson_val = st.number_input(
                        display_name,
                        value=0,
                        min_value=0,
                        step=1,
                        key=f"input_{raw_name}"
                    )
                    user_input[raw_name] = charlson_val
                else:
                    # 其他特征 - 常规数字输入
                    # 设置特定特征的默认值和范围
                    if "admission_age" in raw_name.lower():
                        default = 60.0
                        min_value = 0.0
                        max_value = 120.0
                    elif "sbp" in raw_name.lower():
                        default = 120.0
                        min_value = 0.0
                        max_value = 300.0
                    elif "spo2" in raw_name.lower():
                        default = 98.0
                        min_value = 0.0
                        max_value = 100.0
                    elif "temperature" in raw_name.lower():
                        default = 36.5
                        min_value = 30.0
                        max_value = 45.0
                    elif "urineoutput_24h" in raw_name.lower():
                        default = 1000.0
                    elif "platelet" in raw_name.lower():
                        default = 200.0
                    elif "creatinine" in raw_name.lower():
                        default = 0.8
                    elif "potassium" in raw_name.lower():
                        default = 4.0
                    elif "hemoglobin" in raw_name.lower():
                        default = 12.0
                    elif "apsiii" in raw_name.lower() or "oasis" in raw_name.lower():
                        default = 30.0
                        step = 1.0

                    val = st.number_input(
                        display_name,
                        value=default,
                        min_value=min_value,
                        max_value=max_value,
                        step=step,
                        format="%.3f",
                        key=f"input_{raw_name}"
                    )
                    user_input[raw_name] = val

    submitted = st.form_submit_button("🚀 Predict / 点击预测")

if submitted:
    # 检查输入完整性
    error_msgs = []
    for group_features in feature_groups.values():
        for raw_name, display_name in group_features:
            if raw_name not in user_input or pd.isna(user_input[raw_name]):
                error_msgs.append(f"'{display_name}' requires a valid value")
                error_msgs.append(f"'{display_name}' 需要提供有效值")

    if error_msgs:
        st.error("**Input Errors / 输入错误:**")
        for msg in error_msgs:
            st.error(msg)
    else:
        try:
            # 准备输入数据
            input_data = {}
            for feat in feature_names:
                input_data[feat] = user_input[feat]

            input_df = pd.DataFrame([input_data])
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

# 添加模型信息侧边栏 - 全部使用英文
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"**Algorithm**: CatBoost (SHAP Simplified)")
    st.markdown(f"**Number of Features**: {len(feature_names)}")
    st.markdown(f"**Optimal Threshold**: {optimal_threshold:.4f}")

    st.divider()
    st.markdown("""
    **Development Notes**:  
    This tool was developed based on MIMIC database (version 3.1) research data, using SHAP feature selection method to identify the most important 14 predictors from the 51 original features after data feature engineering, and externally validated using eICU-CRD (version 2.0) data.
    """)

    # 添加研究团队信息
    st.markdown("### Development Team")
    st.caption("- Principal Investigator & Data Analyst: Dr. Jianyuan Liu")
    st.caption("- Clinical Consultant: Prof. Shubin Guo")

    # 添加文件下载功能
    st.divider()
    st.markdown("### Download Resources")
    try:
        with open("shap_simplified_model.cbm", "rb") as file:
            st.download_button(
                label="Download Model",
                data=file,
                file_name="delirium_prediction_model.cbm",
                mime="application/octet-stream"
            )
    except:
        st.warning("Model file not available for download")

    try:
        with open("shap_simplified_model_threshold.json", "rb") as file:
            st.download_button(
                label="Download Config",
                data=file,
                file_name="model_config.json",
                mime="application/json"
            )
    except:

        st.warning("Config file not available for download")
