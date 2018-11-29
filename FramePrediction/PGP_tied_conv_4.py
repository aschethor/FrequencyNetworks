"""
PGP architecture with one layer
optional weight sharing
"""

import torch
import torch.nn as nn
import numpy as np
from util import *


class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100],hidden_size=20,share_U=True,share_W=True):
		super(Model, self).__init__()
		if hidden_size%2 != 0:
			raise ValueError('the parameter "hidden_size" must be even!')
		
		# later, n_input_images could correspond to number of layers...
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		self.hidden_size = hidden_size
		
		# generate learnable modules
		self.u1_t = nn.Conv2d( 1, hidden_size, kernel_size=11, padding=5, bias=False)
		self.w1 = nn.Conv2d( hidden_size, hidden_size, kernel_size=3, padding=1, bias=True)
		
		self.softRELU = nn.Softplus()
		
		self.u1_prime = nn.ConvTranspose2d( hidden_size, 1, kernel_size=11, padding=5, bias=True)
		self.w1_prime_t = nn.ConvTranspose2d( hidden_size, hidden_size, kernel_size=3, padding=1, bias=False)
		
		# share weights
		if share_U:
			self.u1_prime.weight = self.u1_t.weight
		if share_W:
			self.w1_prime_t.weight = self.w1.weight
		
		# generate transformations needed for "complex characteristics"
		I,P,B,R = trafo_matrices(hidden_size)
		self.E1 = torch.cat([I,I],dim=0).cuda()
		self.E2 = torch.cat([I,B],dim=0).cuda()
		self.E3 = torch.cat([I,torch.mm(B,R)],dim=0).cuda()
		self.P1 = P.cuda()
		self.P2 = P.cuda()
		
	def forward(self,x):
		
		x = torch.mean(x,dim=2).unsqueeze(2)
		x_t1 = x[:,self.n_input_images-2,:,:,:]
		x_t2 = x[:,self.n_input_images-1,:,:,:]
		x1 = channel_batch_mm(self.E1,self.u1_t(x_t1))
		x2 = channel_batch_mm(self.E2,self.u1_t(x_t2))
		x = x1*x2
		x = channel_batch_mm(self.P1,x)
		x = self.softRELU(self.w1_prime_t(x, output_size = self.image_shape))
		x = channel_batch_mm(self.E3,x)
		x3 = channel_batch_mm(self.E1,self.u1_t(x_t2))
		x = x*x3
		x = channel_batch_mm(self.P2,x)
		x_t3 = self.u1_prime(x, output_size = self.image_shape)
		return x_t3*torch.ones(1,3,1,1).cuda()
