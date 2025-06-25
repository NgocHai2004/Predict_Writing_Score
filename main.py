import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
# Load data
data = pd.read_csv("StudentScore.xls")  # hoặc Read_File_CSV() nếu bạn có

X = data.drop(columns=["writing score"])
y = data["writing score"]

# Tách train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Định nghĩa cột
num_cols = ["math score", "reading score"]
ord_cols = ["parental level of education"]
nom_cols = ["gender", "race/ethnicity", "lunch", "test preparation course"]

# Giá trị ordinal theo thứ tự
ord_levels = [["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]]

# Tạo transformer cho từng loại biến
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('ord', OrdinalEncoder(categories=ord_levels), ord_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore'), nom_cols)
])

# Tạo pipeline tổng: tiền xử lý + model
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "diabetes_pipeline.pkl")

# Dự đoán
# y_pred = pipeline.predict(X_test)


# # Đánh giá
# # print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
# # print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
# # print(f"R^2: {r2_score(y_test, y_pred):.2f}")

# # Dự đoán thử 1 dòng mới
# sample = pd.DataFrame([{
#     "gender": "female",
#     "race/ethnicity": "group B",
#     "parental level of education": "bachelor's degree",
#     "lunch": "standard",
#     "test preparation course": "none",
#     "math score": 52,
#     "reading score": 72
# }])

# sample_pred = pipeline.predict(sample)
# print(f"Dự đoán cho mẫu mới: {sample_pred[0]:.0f}")
