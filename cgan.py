#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from Generator import Generator
from Discriminator import Discriminator

def seed_everything(seed=69):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

##################### Inputs #####################

# Batch size for training
batch_size = 32

# All images will be resized to this size using transformer.
image_size = 28*28

# Size of z latent vector (i.e. size of generator input)
nz = 64

# Size of feature maps in generator
ngf = 256

# Size of feature maps in discriminator
ndf = 256

# Number of training epochs - change this!
num_epochs = 0

# Learning rate for optimizers
lr = 0.0002

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

##################### Data #####################

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = dset.MNIST(root='./data/', train=True, transform=transform, download=True)

# Data loader
dataloader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2).cpu(),(1,2,0)))
plt.show()

##################### Weight Initialization #####################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('conv')
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print('batch')
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


##################### Initialize Generator and Discriminator #####################

netG = Generator(ngpu, nz, ngf, image_size).to(device)
netG.apply(weights_init)

netD = Discriminator(ngpu, ndf, image_size).to(device)
netD.apply(weights_init)

print(netG)
print(netD)

##################### Loss Functions and Optimizers #####################

# Initialize BCELoss function
criterion = nn.BCELoss()

fixed_noise = torch.randn(batch_size, nz, device=device)

# Initialize Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)


##################### Training #####################

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

patience = 5
# current number of epochs, where validation loss didn't increase
p = 0
# whether training should be stopped
stop = False

print('Starting Training Loop...')
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        # Establish convention for real and fake labels
        # Randomly recompute each batch for "soft label"
        real_label = random.uniform(0.9, 1.0)
        fake_label = random.uniform(0.0, 0.1)

        ################ Update disriminator ################

        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].view(batch_size,-1).to(device)
        real_cpu_labels = data[1]
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu, real_cpu_labels).view(batch_size, -1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size))
        # Generate fake image batch with G
        fake = netG(noise, fake_labels)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), fake_labels).view(batch_size, -1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ################ Update generator ################

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, fake_labels).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise, torch.LongTensor(np.random.randint(0,10, batch_size))).detach().cpu()
            img_list.append(vutils.make_grid(np.reshape(fake,(32,1,28,28)), padding=2))

        iters += 1

    # Save output images every 10 epochs
    if epoch % 10 == 0:
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.axis("off")
        plt.savefig('output/epoch-{}'.format(epoch))

# Save network checkpoints
torch.save(netG.state_dict(), 'generator.ckpt')
torch.save(netD.state_dict(), 'discriminator.ckpt')

##################### Results #####################

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
