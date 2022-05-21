from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch
from dataloader.Augmented_dataloader import  LoadCsvData
from Models.New_Framework import GAT_LSTM
import numpy as np
import sklearn.metrics as metrics
class Solver():

    def __init__(self,
                 optimizer,
                 epoch,
                 model,
                 weight_decay=0,
                 batch_size=1,
                 lr=1e-3,
                 sequence_size = 10,
                 train_part=0.7,
                 file_path = "../data/code-red.csv"
                 ):

        assert model is not None
        self.model = model
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        else:
            print("invalid optimizer name!")
            exit(1)

        assert epoch > 0
        self.epoch = epoch
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.file_path = file_path
        CsvData = LoadCsvData(self.file_path, windows_size=self.sequence_size)

        Train_size = int(len(CsvData)*train_part)
        Test_size = len(CsvData)-Train_size
        train_dataset,test_dataset = torch.utils.data.random_split(CsvData,[Train_size,Test_size])
        self.TrainDataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.TestDataloader = DataLoader(
            dataset = test_dataset,
            batch_size=1,
            shuffle=True
        )

        self.loss = nn.CrossEntropyLoss()

    def train(self):
        best_acc = 0
        for cur_epoch in range(self.epoch):
            for i, batch in enumerate(self.TrainDataloader):
                input_feature,label = batch
                label = torch.squeeze(label,0)
                label = label.cuda()
                input_feature = input_feature.cuda()
                input_feature = input_feature.squeeze(0)
                out = self.model(input_feature)
                out = torch.unsqueeze(out,0)
                loss = self.loss(out,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i%100 == 0:
                    print("epoch:%d | Iter:%d | loss:%f"%(cur_epoch,i,loss.item()))
            acc,_,_ = self.valid()
            if acc > best_acc:
                best_acc = acc
                print("save best pth...")
                self.SaveBestCkpt()
            self.model.train()
        pass

    def valid(self):
        avg_loss = 0
        total_batch = 0
        correct_class = 0
        true_label = list()
        predicted_label = list()
        print("validating-------------------")
        for i, batch in enumerate(self.TestDataloader):
            self.model.eval()
            input_feature, label = batch
            input_feature = input_feature.cuda()
            label = label.cuda()
            label = torch.squeeze(label, 0)
            input_feature = input_feature.squeeze(0)
            out = self.model(input_feature)
            out = torch.unsqueeze(out, 0)
            with torch.no_grad():
                out_label = out.detach().cpu().numpy()
                out_label = np.argmax(out_label,axis=1)[0]
                true_label.append(label.detach().cpu().numpy()[0])
                predicted_label.append(out_label)
                if out_label == label:
                    correct_class +=1
                loss = self.loss(out, label)
                avg_loss+=loss.item()
            total_batch+=1
        avg_loss/= total_batch
        accuracy = correct_class/total_batch
        print("avg_loss:",avg_loss)
        print("accuracy:",accuracy)
        return accuracy,true_label,predicted_label
        pass
    def SaveBestCkpt(self):
        torch.save(self.model, 'pretrained_models/nimda/best_param.pth')


    def LoadCkpt(self,path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt.state_dict())

    def Metrics_eval(self,model_path):
        self.LoadCkpt(model_path)
        acc,true_label,predicted_label = self.valid()
        f1_score = metrics.f1_score(true_label,predicted_label)
        acc_score = metrics.accuracy_score(true_label,predicted_label)
        recall_score = metrics.recall_score(true_label,predicted_label)
        precision_score = metrics.precision_score(true_label,predicted_label)
        print("f1_score:",f1_score)
        print("acc_score:",acc_score)
        print("recall_score:",recall_score)
        print("precision_score:",precision_score)




if __name__ == "__main__":

    feature_size = 46

    Model = GAT_LSTM(
        window_size=30
    )
    Model = Model.cuda()
    # model = MyMLP(
    #     feature_size=feature_size,
    #     hidden_size=100,
    #     output_class=2
    # )


    RNNSolver = Solver(
        model=Model,
        epoch=50,
        optimizer='Adam',
        weight_decay=0,
        sequence_size=30,
        file_path= "./data/nimda.csv",
        train_part=0.6
    )
    RNNSolver.Metrics_eval('./pretrained_models/nimda/best_param.pth')
    #RNNSolver.LoadCkpt('./pretrained_models/nimda/best_param.pth')
    # RNNSolver.train()
    # RNNSolver.valid()


