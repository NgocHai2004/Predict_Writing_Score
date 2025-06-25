import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load("diabetes_pipeline.pkl")

st.title("🎓 Dự đoán điểm Writing Score")

# Form nhập dữ liệu
gender = st.selectbox("Giới tính", ["male", "female"])
race = st.selectbox("Nhóm chủng tộc", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Trình độ phụ huynh", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Loại bữa ăn", ["standard", "free/reduced"])
prep_course = st.selectbox("Khóa ôn luyện", ["none", "completed"])
math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)

# Khi bấm nút Dự đoán
if st.button("Dự đoán điểm Writing"):
    # Tạo DataFrame từ dữ liệu nhập
    input_data = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch,
        "test preparation course": prep_course,
        "math score": math_score,
        "reading score": reading_score
    }])

    # Dự đoán
    prediction = model.predict(input_data)
    st.success(f"🎯 Điểm writing dự đoán: {prediction[0]:.0f}")
