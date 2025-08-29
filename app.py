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
st.title("Delirium Prediction Tool for Sepsis Patients in ICU")
st.title("ICU脓毒症患者谵妄预测工具")

st.markdown(f"""
**Instructions**:  
Please enter the patient information below. The system will predict the probability of delirium occurring within 7 days after ICU admission for sepsis patients based on the externally validated SHAP interpretable model.  
All measured variables should use average values from the first 24 hours after ICU admission (except GCS which uses the lowest value within the first 24 hours).

**说明**:  
请输入以下的患者信息，系统将基于经外部验证的SHAP可解释性模型预测ICU脓毒症患者在ICU入科7天内发生谵妄的概率。  
以下所有需测量的变量均需使用入ICU后第1天内的平均值数据（gcs为取入ICU1天内最低值）。
""")

# 定义特征分组和显示名称
feature_groups = {
    "Basic Information / 基础情况": [
        "admission_age/年龄 (years)",
        "hypertension/高血压"
    ],
    "Bedside Signs / 床旁体征": [
        "sbp/收缩压 (mmHg)",
        "spo2/血氧饱和度 (%)",
        "temperature/体温 (℃)",
        "urineoutput_24h/尿量 (ml)"
    ],
    "Laboratory Tests / 实验室检查": [
        "platelet/血小板计数 (K/μL)",
        "creatinine/血清肌酐 (mg/dL, 1mg/dL=88.4μmol/L)",
        "potassium/钾离子浓度 (mmol/L)",
        "hemoglobin/血红蛋白 (g/dL)"
    ],
    "Clinical Scores / 临床评分": [
        "charlson_comorbidity_index/Charlson共病指数",
        "gcs_min/最低格拉斯哥昏迷评分",
        "apsiii/急性生理评分III",
        "oasis/牛津急性疾病严重程度评分"
    ]
}

# 构建输入表单
user_input = {}
with st.form("input_form"):
    # 为每个特征组创建部分
    for group_name, features in feature_groups.items():
        st.subheader(group_name)
        cols = st.columns(2)  # 每组使用两列布局

        for i, feat_info in enumerate(features):
            col_idx = i % 2
            with cols[col_idx]:
                # 解析特征信息
                parts = feat_info.split("/")
                eng_name = parts[0].strip()
                chn_name = parts[1].split("(")[0].strip() if len(parts) > 1 else ""
                unit_info = feat_info.split("(")[1].split(")")[0] if "(" in feat_info else ""

                # 设置显示名称
                display_name = f"{eng_name} / {chn_name}"
                if unit_info:
                    display_name += f" ({unit_info})"

                # 设置默认值和约束
                default = 0.0
                min_value = 0.0
                max_value = None
                step = 0.1

                # 特殊处理某些特征
                if "hypertension" in eng_name.lower():
                    # 高血压 - 单选按钮
                    st.markdown(f"**{display_name}**")
                    hypertension_val = st.radio(
                        "",
                        options=[0, 1],
                        index=0,
                        horizontal=True,
                        key=f"input_{eng_name}"
                    )
                    user_input[eng_name] = hypertension_val
                elif "gcs_min" in eng_name.lower():
                    # GCS评分 - 滑块
                    st.markdown(f"**{display_name}**")
                    gcs_val = st.slider(
                        "",
                        min_value=3,
                        max_value=15,
                        value=15,
                        step=1,
                        key=f"input_{eng_name}"
                    )
                    user_input[eng_name] = gcs_val
                elif "charlson_comorbidity_index" in eng_name.lower():
                    # Charlson共病指数 - 整数输入
                    charlson_val = st.number_input(
                        display_name,
                        value=0,
                        min_value=0,
                        step=1,
                        key=f"input_{eng_name}"
                    )
                    user_input[eng_name] = charlson_val
                else:
                    # 其他特征 - 常规数字输入
                    # 设置特定特征的默认值和范围
                    if "admission_age" in eng_name.lower():
                        default = 60.0
                        min_value = 0.0
                        max_value = 120.0
                    elif "sbp" in eng_name.lower():
                        default = 120.0
                        min_value = 0.0
                        max_value = 300.0
                    elif "spo2" in eng_name.lower():
                        default = 98.0
                        min_value = 0.0
                        max_value = 100.0
                    elif "temperature" in eng_name.lower():
                        default = 36.5
                        min_value = 30.0
                        max_value = 45.0
                    elif "urineoutput_24h" in eng_name.lower():
                        default = 1000.0
                    elif "platelet" in eng_name.lower():
                        default = 200.0
                    elif "creatinine" in eng_name.lower():
                        default = 0.8
                    elif "potassium" in eng_name.lower():
                        default = 4.0
                    elif "hemoglobin" in eng_name.lower():
                        default = 12.0
                    elif "apsiii" in eng_name.lower() or "oasis" in eng_name.lower():
                        default = 30.0
                        step = 1.0

                    val = st.number_input(
                        display_name,
                        value=default,
                        min_value=min_value,
                        max_value=max_value,
                        step=step,
                        format="%.3f",
                        key=f"input_{eng_name}"
                    )
                    user_input[eng_name] = val

    submitted = st.form_submit_button("🚀 Predict / 点击预测")

if submitted:
    # 检查输入完整性
    error_msgs = []
    for group_features in feature_groups.values():
        for feat_info in group_features:
            eng_name = feat_info.split("/")[0].strip()
            if eng_name not in user_input or pd.isna(user_input[eng_name]):
                error_msgs.append(f"'{eng_name}' requires a valid value")
                error_msgs.append(f"'{eng_name}' 需要提供有效值")

    if error_msgs:
        st.error("**Input Errors / 输入错误:**")
        for msg in error_msgs:
            st.error(msg)
    else:
        try:
            # 准备输入数据
            input_data = {}
            for feat in feature_names:
                # 找到对应的英文名称
                for group_features in feature_groups.values():
                    for feat_info in group_features:
                        eng_name = feat_info.split("/")[0].strip()
                        if feat == eng_name:
                            input_data[feat] = user_input[eng_name]
                            break

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

# 添加模型信息侧边栏
with st.sidebar:
    st.header("ℹ️ 模型信息")
    st.markdown(f"**算法**: CatBoost (SHAP简化版)")
    st.markdown(f"**特征数**: {len(feature_names)}")
    st.markdown(f"**最佳阈值**: {optimal_threshold:.4f}")

    st.divider()
    st.markdown("""
    **开发说明**:  
    This tool was developed based on MIMIC database (version 3.1) research data, using SHAP feature selection method to identify the most important 14 predictors from the 51 original features after data feature engineering, and externally validated using eICU-CRD (version 2.0) data.

    本工具基于MIMIC数据库（version 3.1）研究数据开发，采用SHAP特征选择方法，从经过数据特征工程后的51个原始特征中筛选出最重要的14个预测因子，并且通过eICU-CRD（version 2.0）的数据进行了外部验证。
    """)

    # 添加研究团队信息
    st.markdown("### Development Team / 开发团队")
    st.caption("- Principal Investigator & Data Analyst: Dr. Jianyuan Liu")
    st.caption("- Clinical Consultant: Prof. Shubin Guo")

    # 添加文件下载功能
    st.divider()
    st.markdown("### 资源下载")
    try:
        with open("shap_simplified_model.cbm", "rb") as file:
            st.download_button(
                label="下载模型",
                data=file,
                file_name="delirium_prediction_model.cbm",
                mime="application/octet-stream"
            )
    except:
        st.warning("模型文件不可下载")

    try:
        with open("shap_simplified_model_threshold.json", "rb") as file:
            st.download_button(
                label="下载配置",
                data=file,
                file_name="model_config.json",
                mime="application/json"
            )
    except:
        st.warning("配置文件不可下载")