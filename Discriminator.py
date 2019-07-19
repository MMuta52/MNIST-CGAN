import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, image_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.label = nn.Embedding(10,10) # use embedding layer to represent label in network
        self.main = nn.Sequential(
            nn.Linear(image_size + 10, ndf),
            nn.LeakyReLU(0.2,),
            nn.Linear(ndf, ndf),
            nn.LeakyReLU(0.2),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        c = self.label(labels)
        x = torch.cat([input, c], 1)
        return self.main(x)
