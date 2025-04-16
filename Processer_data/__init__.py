from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Lớp cơ sở xử lý dữ liệu
class Processer_Data:
    def __init__(self, x):
        '''x: Dữ liệu đầu vào cần xử lý (DataFrame hoặc Series)'''
        self.x = x

# Lớp xử lý dữ liệu số (Numerical)
class Processer_Numerical(Processer_Data):
    def __init__(self, x):
        super().__init__(x)  # Khởi tạo lớp cha

    def type_process_num(self):
        '''Trả về pipeline xử lý dữ liệu số: điền giá trị thiếu và chuẩn hóa'''
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Điền giá trị thiếu bằng giá trị trung vị
            ("scaler", StandardScaler())  # Chuẩn hóa dữ liệu, biến đổi sao cho có giá trị trung bình = 0 và độ lệch chuẩn = 1
        ])

    def process_num(self):
        return self.type_process_num().fit_transform(self.x)

# Lớp xử lý dữ liệu ordinal (thứ bậc)
class Processer_Ordinal(Processer_Data):
    def __init__(self, x, level):
        '''Các cấp độ (level) của dữ liệu ordinal'''
        super().__init__(x)  # Khởi tạo lớp cha
        self.level = level  

    def type_process_ord(self):
        '''Trả về pipeline xử lý dữ liệu ordinal: điền giá trị thiếu và mã hóa thứ tự'''
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất
            ("encoder", OrdinalEncoder(categories=self.level))  # Mã hóa thứ tự, sử dụng các cấp độ đã cho
        ])
    
    def process_ord(self):
        return self.type_process_ord().fit_transform(self.x)


# Lớp xử lý dữ liệu nominal (danh mục không thứ tự)
class Processer_Nominal(Processer_Data):
    def __init__(self, x):
        super().__init__(x)  # Khởi tạo lớp cha

    def type_process_nom(self):
        '''Trả về pipeline xử lý dữ liệu nominal: điền giá trị thiếu và mã hóa One-Hot'''
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))  # Mã hóa One-Hot, không tạo cột cho giá trị chưa thấy
        ])
    def process_nom(self):
        return self.type_process_nom().fit_transform(self.x)
    

# Lớp xử lý nhãn mục tiêu (target variable)
class Processer_Target(Processer_Data):
    def __init__(self, x):
        super().__init__(x)  # Khởi tạo lớp cha

    def type_process_tar(self):
        '''Trả về pipeline xử lý nhãn mục tiêu: điền giá trị thiếu và mã hóa nhãn'''
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất
            ("encoder", LabelEncoder())  # Mã hóa nhãn thành các số nguyên (thích hợp cho các bài toán phân loại)
        ])
    
def data_use_model(x_train,y_train,model):
    model = LinearRegression()
    return model.fit(x_train,y_train)


