import torch.nn as nn
import torch


class QNetwork(nn.Module):

    def __init__(self, ob_dim, ac_dim, hidden_dim=25):
        super().__init__()
        self.linear1 = #TODO
        self.linear2 = #TODO
        self.linear3 = #TODO

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x

def num_params(module):
    antall = 0
    for params in module.parameters():
        antall += params.numel()
    return antall


if __name__ == '__main__':
    network = QNetwork(4, 2)
    print(num_params(network))
    print(network(torch.zeros(4)))
