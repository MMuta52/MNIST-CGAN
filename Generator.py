import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, image_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.label = nn.Embedding(10,10) # use embedding layer to represent label in network
        self.main = nn.Sequential(
            nn.Linear(nz + 10, ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf),
            nn.ReLU(),
            nn.Linear(ngf, image_size),
            nn.Tanh()
        )

    def forward(self, input, labels):
        c = self.label(labels)
        x = torch.cat([input, c], 1)
        return self.main(x)
