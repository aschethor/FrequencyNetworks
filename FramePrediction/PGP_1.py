"""
Conv - PGP architecture with one layer
no weight sharing
"""

import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100]):
		super(Model, self).__init__()
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		
		self.u1 = nn.Conv2d( 3, 20, kernel_size=5,padding = 2)
		self.v1 = nn.Conv2d( 3, 20, kernel_size=5,padding = 2)
		self.w1 = nn.Conv2d( 20, 20, kernel_size=5,padding = 2)
		
		self.u1_prime = nn.Conv2d( 3, 20, kernel_size=5,padding = 2)
		self.v1_prime_t = nn.ConvTranspose2d( 20, 3, kernel_size=5, padding=2)
		self.w1_prime_t = nn.ConvTranspose2d( 20, 20, kernel_size=5, padding=2)
		
		
	def forward(self,x):
		x_t1 = x[:,self.n_input_images-2,:,:,:]
		x_t2 = x[:,self.n_input_images-1,:,:,:]
		x1 = self.u1(x_t1)
		x2 = self.v1(x_t2)
		x = x1*x2
		x = torch.sigmoid(self.w1(x))
		x = self.w1_prime_t(x, output_size = self.image_shape)
		x3 = self.u1_prime(x_t2)
		x = x*x3
		x_t3 = self.v1_prime_t(x, output_size = self.image_shape)
		return x_t3

