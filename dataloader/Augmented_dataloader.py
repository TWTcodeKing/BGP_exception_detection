import numpy as np
import pandas
import pandas as pd
from utils.STL import STL_Process
from torch.utils.data import Dataset,DataLoader
import torch

def load_data(data_path = '../data/code-red.csv'):
    DataFrame = pandas.read_csv(data_path)
    feature_names = DataFrame.columns
    features_5k = STL_Process(DataFrame[feature_names[0]])
    print("loading data into memory...")
    for i in range(1,len(feature_names)-1):
        features_5k_tmp = STL_Process(DataFrame[feature_names[i]])
        features_5k = np.hstack([features_5k,features_5k_tmp])
    labels = DataFrame["class"].values
    return features_5k,labels

class LoadCsvData(Dataset):
    def __init__(self,file_path,windows_size):
        super(LoadCsvData,self).__init__()
        self.features_5k,self.labels = load_data(file_path)
        self.labels = self.labels[:,np.newaxis]
        self.vector_and_label = None
        self.window_size = windows_size

    def __getitem__(self, item):
        vector,label = self.Window_slicing(item)
        assert vector is not None
        assert label is not None
        return vector,label
        pass

    def __len__(self):
        return (len(self.labels))-self.window_size


    def Window_slicing(self,pointer):
        features_5k = self.features_5k[pointer:pointer+self.window_size,:]
        label = self.labels[pointer:pointer+self.window_size]
        single_label = label[-1]
        features_5k = torch.FloatTensor(features_5k)
        single_label = torch.LongTensor(single_label)
        return features_5k,single_label
        pass

if __name__ == "__main__":
    ds = LoadCsvData(
        file_path='../data/code-red.csv',
        windows_size=30
    )
    MyLoader = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle= True
    )
    for i, batch in enumerate(MyLoader):
        feature,label = batch
        print(feature.shape)