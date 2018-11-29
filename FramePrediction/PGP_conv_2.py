"""
PGP architecture with one layer
optional weight sharing
"""

import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100],hidden_size=20,share_U=True,share_V=True,share_W=True):
		super(Model, self).__init__()
		#later, n_input_images could correspond to number of layers...
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		self.hidden_size = hidden_size
		
		self.u1 = nn.Conv2d( 1, hidden_size, kernel_size=11,padding = 5,bias=False)
		self.v1 = nn.Conv2d( 1, hidden_size, kernel_size=11,padding = 5,bias=False)
		self.w1 = nn.Conv2d( hidden_size, hidden_size, kernel_size=3,padding = 1,bias=False)
		
		self.u1_prime = nn.Conv2d( 1, hidden_size, kernel_size=11,padding = 5,bias=False)
		self.v1_prime_t = nn.ConvTranspose2d( hidden_size, 1, kernel_size=11, padding=5,bias=False)
		self.w1_prime_t = nn.ConvTranspose2d( hidden_size, hidden_size, kernel_size=3, padding=1,bias=False)
		
		if share_U:
			self.u1_prime.weight = self.u1.weight
		if share_V:
			self.v1_prime_t.weight = self.v1.weight
		if share_W:
			self.w1_prime_t.weight = self.w1.weight
		
		
	def forward(self,x):
		x = torch.mean(x,dim=2).unsqueeze(2)
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
		return x_t3*torch.ones(1,3,1,1).cuda()

