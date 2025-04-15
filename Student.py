import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


data = pd.read_csv("StudentScore.xls")
# report = ProfileReport(data,title ="Profiling Report")
# report.to_file("report.html")
target = "writing score"
x = data.drop([target],axis=1)
y = data[target]
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

num = ["reading score","reading score"]
nom = ["race/ethnicity","test preparation course","lunch","gender"]
ord = ["parental level of education"]
set = [["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]]
num_tranformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("encoder",StandardScaler())
])
ord_tranformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",(OrdinalEncoder(categories=set)))])
nom_tranformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
proprecessor = ColumnTransformer(transformers=[
    ("num",num_tranformer,num),
    ("ord",ord_tranformer,ord),
    ("nom",nom_tranformer,nom)
])
xtrain = proprecessor.fit_transform(x_train)
xtest = proprecessor.fit_transform(x_test)
df = pd.DataFrame(xtrain)

# # Sắp xếp theo cột chỉ mục (index = 1) giảm dần
# sorted_df = df.sort_values(by=1, ascending=False)
print(df.iloc[:3])


