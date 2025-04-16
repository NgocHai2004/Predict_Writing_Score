from Read_report import Read_File_CSV
from Split_data import Split_Data 
from Processer_data import Processer_Data,Processer_Nominal,Processer_Numerical,Processer_Ordinal,data_use_model
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import pandas as pd

def main():
    path = os.path.abspath("StudentScore.xls")
    data = Read_File_CSV(path).read()
    x = data.drop(columns=["writing score"])
    y = data["writing score"]
    x_train,x_test,y_train,y_test = Split_Data(x,y).split_train_test()
    
    nom = ["gender","race/ethnicity","lunch","test preparation course"]
    num = ["math score","reading score"]
    ord = ["parental level of education"]
    level = [["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]]
    
    # Xử lí dữ liệu phần x_train
    num_processor_train = Processer_Numerical(x_train[num]).process_num()
    ord_processor_train = Processer_Ordinal(x_train[ord],level).process_ord()
    nom_processor_train = Processer_Nominal(x_train[nom]).process_nom()
    x_train = np.hstack((ord_processor_train,num_processor_train,nom_processor_train))

    # sử lí dữ liệu phần x_test
    num_processor_test = Processer_Numerical(x_test[num]).process_num()
    ord_processor_test = Processer_Ordinal(x_test[ord],level).process_ord()
    nom_processor_test = Processer_Nominal(x_test[nom]).process_nom()
    x_test = np.hstack((ord_processor_test,num_processor_test,nom_processor_test))


    model = data_use_model(x_train,y_train,LinearRegression())
    y_pred = model.predict(x_test)
    # for i,j in zip(y_pred,y_test):
    #     print(f"{i:.0f}------{j}")
    
    t = ["female","group B","bachelor's degree","standard","none",52,72]
    te = pd.DataFrame([t],columns=["gender","race/ethnicity","parental level of education","lunch","test preparation course","math score","reading score"])
    print(te)
   
    
    


main()