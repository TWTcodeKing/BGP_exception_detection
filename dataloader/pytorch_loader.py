import pandas
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
class LoadCsvData(Dataset):
    def __init__(self,file_path,windows_size):
        super(LoadCsvData,self).__init__()
        self.csvfile = file_path
        self.vector_and_label = None
        self.windows_size = windows_size
        self.Isread = False
        self.Len = 0
        self.current_pointer = 0
        self.InitDataLoader()
        self.positive_sample =0

    def __getitem__(self, item):
        vector,label = self.CsvParser(item)
        assert vector is not None
        assert label is not None
        return vector,label
        pass

    def __len__(self):
        if self.Isread:
            return self.Len
        else:
            temp = pandas.read_csv(self.csvfile)
            self.Len = temp.values.shape[0]-self.windows_size
            self.Isread = True
            return self.Len


    def CsvParser(self,file_pointer):
        Embedding_matrix = list()
        labels = list()
        self.vector_and_label.seek(0)
        for i in range(file_pointer):
            self.vector_and_label.readline()
        if (file_pointer == 0):
            self.vector_and_label.readline()
        self.current_pointer += self.windows_size
        if self.current_pointer >= self.__len__()*5:
            self.vector_and_label.seek(0)
            self.vector_and_label.readline()
            self.current_pointer = 0
        for i in range(self.windows_size):
            line = self.vector_and_label.readline()
            line = line.strip("\n")
            line = line.strip("\r")
            vector_and_label = line.split(",")
            vector = list(map(int,vector_and_label[:-1]))
            label  = int(vector_and_label[-1])
            Embedding_matrix.append(vector)
            labels.append(label)
        Embedding_matrix = torch.Tensor(np.asarray(Embedding_matrix))
        if labels[0] == 1 and labels[self.windows_size-1] == 0:
            return Embedding_matrix,labels[-1]
        elif labels[0] == 0 and labels[self.windows_size-1] == 1:
            return Embedding_matrix,labels[-1]
        else:
            return Embedding_matrix,labels[0]
        pass

    def InitDataLoader(self):
        self.vector_and_label = open(self.csvfile,'r',encoding='utf-8',newline='')
        self.vector_and_label.readline()

        pass




if __name__ == "__main__":
    file_path = "../data/dataset_multi_aws-leak_15547_1_rrc04.csv"
    CsvData = LoadCsvData(file_path,windows_size=5)
    MyLoader = DataLoader(
        dataset=CsvData,
        batch_size=1,
        shuffle= True
    )
    for i, batch in enumerate(MyLoader):
        feature,label = batch
