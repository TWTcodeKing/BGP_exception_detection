import torch
from torch import nn

class MyRNN(nn.Module):
    def __init__(self,feature_size,hidden_size,num_layers):
        super(MyRNN,self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.norm = nn.LayerNorm(
            normalized_shape=self.hidden_size
        )
        self.rnn = nn.RNN(
            input_size= feature_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.out = torch.nn.Linear(self.hidden_size,2)





    def forward(self,x):
        x = x.view(x.size(1), x.size(0), -1)
        h_state = torch.zeros(self.num_layers,x.size(1),self.hidden_size).cuda()
        out_x,h_state = self.rnn(x,h_state)
        #out_x = self.norm(out_x)
        out = out_x[-1]
        out = self.out(out)
        return out

if __name__ == "__main__":
    model = MyRNN(feature_size=10,
                  hidden_size=20,
                  num_layers=1)
    input = torch.rand(5,1,10)

    y = model(input)


