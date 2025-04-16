from sklearn.model_selection import train_test_split
import pandas as pd


class Split_Data:
    def __init__(self,x,y):
        '''data:Dữ liệu truyền vào từ file CSV'''
        self.x = x
        self.y = y

    
    def split_train_test(self):
        return train_test_split(self.x,self.y,test_size=0.2,random_state=42)

    


        

        


