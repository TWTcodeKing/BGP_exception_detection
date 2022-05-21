import torch
from Models.New_Framework import GAT_LSTM
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Model = GAT_LSTM(
        window_size=30
    )
    ckpt = torch.load('../pretrained_models/code-red/best_param.pth')
    Model.load_state_dict(ckpt.state_dict())
    adj_temp = Model.adj_temporal.detach().numpy()
    adj_feat = Model.adj_features.detach().numpy()

    plt.matshow(adj_feat)
    plt.show()