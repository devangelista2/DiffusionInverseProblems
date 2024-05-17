import torch
from models.GAN.DCGAN import *


z = torch.randn((1, 1, 64, 64))
N, c, h, w = z.shape

G = Generator(input_shape=(c, h, w))
x = G(z)

D = Discriminator(input_shape=(c, h, w))
c = D(x)
