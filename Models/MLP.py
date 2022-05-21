import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self,feature_size,hidden_size,output_class,window_size):
        super(MyMLP,self).__init__()
        self.feature_size = feature_size
        self.hiddeng_size = hidden_size
        self.output_class = output_class
        self.window_size = window_size
        self.Dense = nn.Sequential(
            nn.Linear(
                in_features=self.feature_size,
                out_features=self.hiddeng_size
                      ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hiddeng_size,
                out_features=self.hiddeng_size//2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hiddeng_size//2,
                out_features=self.hiddeng_size//2,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=self.hiddeng_size//2*self.window_size,
                out_features=self.output_class
            )
        )

    def forward(self,x):
        return self.Dense(x)

if __name__ == "__main__":
    x = torch.rand(1,1,47)
    model = MyMLP(
        feature_size=47,
        hidden_size=100,
        output_class=2
    )
    y = model(x)
    print(y.shape)
