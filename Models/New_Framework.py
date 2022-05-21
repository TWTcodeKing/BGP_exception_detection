import torch
from torch import nn
from Models.GAT import GAT
import numpy as np
class GAT_LSTM(nn.Module):
    def __init__(self,
                 nhid=8,
                 dropout=0.6,
                 alpha=0.2,
                 nheads=8,
                 feature_size = 230,
                 hidden_size= 750,
                 num_layers = 1,
                 window_size=30):
        super(GAT_LSTM,self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.Temporal_GAT = GAT(
                    nfeat = 230,
                    nhid = nhid,
                    nclass = 230 ,
                    dropout = dropout,
                    alpha = alpha,
                    nheads = nheads
                    )

        self.Feature_based_GAT = GAT(
                    nfeat = self.window_size,
                    nhid = nhid,
                    nclass = self.window_size ,
                    dropout = dropout,
                    alpha = alpha,
                    nheads = nheads
                    )
        self.LSTM = nn.LSTM(
            input_size= self.feature_size,
            hidden_size= self.hidden_size,
            num_layers= self.num_layers,
            dropout=0.2
        )
        self.act_lstm = torch.nn.Tanh()
        self.adj_temporal = nn.Parameter(torch.ones(self.window_size,self.window_size))
        nn.init.xavier_uniform_(self.adj_temporal.data, gain=1.414)
        self.adj_features = nn.Parameter(torch.ones(self.feature_size,self.feature_size))
        nn.init.xavier_uniform_(self.adj_features.data,gain=1.414)
        self.norm = nn.LayerNorm(
            normalized_shape=self.hidden_size
        )
        self.out = torch.nn.Linear(self.hidden_size,2)

    def forward(self,x):
        h_Time = self.Temporal_GAT(x,self.adj_temporal)*0.5
        h_init = x.clone()
        h_feat = self.Feature_based_GAT(x.permute(1,0),self.adj_features).permute(1,0)*0.5
        # New_features = torch.cat([h_Time,h_init,h_feat],dim=1).unsqueeze(1)
        New_features = torch.stack([h_Time,h_init,h_feat])
        h_state = torch.randn(self.num_layers, New_features.size(1), self.hidden_size).cuda()
        c_state = torch.randn(self.num_layers, New_features.size(1), self.hidden_size).cuda()
        out_x, h_state = self.LSTM(New_features, (h_state, c_state))
        out_x = self.act_lstm(self.norm(out_x))
        out_x = out_x[1,:,:]
        out = out_x[-1]
        out = self.out(out)
        #finally understand!
        return out
        pass


if __name__ == "__main__":
    Model = GAT_LSTM()
    x= torch.rand(30,230).cuda()
    Model = Model.cuda()
    y = Model(x)
    print(y)
    # one_hot = torch.argmax(y,dim=1)
    # anomous_sample = len(torch.where(one_hot==1)[0])
    # unanomous_sample = len(torch.where(one_hot==0)[0])
    # print(anomous_sample,unanomous_sample)