import pandas as pd
# from ydata_profiling import ProfileReport
class Read_File_CSV:
    def __init__(self,path):
        '''path: dường dẫn file pdf'''
        self.path = path

    def read(self):
        return pd.read_csv(self.path)
    
    # def report(self):
    #     data = self.read()
    #     report = ProfileReport(data, title="Profiling Report")
    #     report.to_file("report.html")


    

        

