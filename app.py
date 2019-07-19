import torch.utils.data
import numpy as np
from PIL import Image
from Generator import Generator
from Discriminator import Discriminator

ngpu = 0
nz = 64
ngf = 256
ndf = 256
image_size = 28*28

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = Generator(ngpu, nz, ngf, image_size)
netG.load_state_dict(torch.load('generator.ckpt'))
netD = Discriminator(ngpu, ndf, image_size)
netD.load_state_dict(torch.load('discriminator.ckpt'))

def number_to_digits(n):
    return [int(i) for i in str(n)]

def generate_digit(n, quality):
    q = 0
    while q < quality:
        noise = torch.randn(1, nz, device=device)
        fake = netG(noise, torch.LongTensor(np.array([n]))).detach().cpu()
        output = netD(fake, torch.LongTensor(np.array([n]))).view(-1)
        q = output.item()
    return np.reshape(fake, (int(np.sqrt(image_size)), int(np.sqrt(image_size))))

def draw_number(number):

    digits = number_to_digits(number)

    all_numbers = np.empty((int(np.sqrt(image_size)), 0))

    for i in range(len(digits)):
        fake = generate_digit(digits[i], quality=0.85)
        fake = fake * 0.5 + 0.5  # Brighten image

        if i == 0 and len(digits) > 1:
            fake = fake[:,:-2]  # Make room to the right
        else:
            if i < len(digits) - 1:
                fake = fake[:,4:-4]  # Make room to the left and right
            else:
                fake = fake[:,2:]  # Make room to the left

        all_numbers = np.hstack((all_numbers, fake))

    img = Image.fromarray(all_numbers * 256)
    img.show()

test = [6, 9, 69, 6969]

for n in test:
    draw_number(n)
