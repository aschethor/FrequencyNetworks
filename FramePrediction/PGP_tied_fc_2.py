"""
PGP architecture with one layer
optional weight sharing
"""

import sys
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')

from CustomLayers.LinearTranspose import LinearTranspose
import torch
import torch.nn as nn
import numpy as np
from util import *


class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100],hidden_size=1000,share_U=True,share_W=True):
		super(Model, self).__init__()
		if hidden_size%2 != 0:
			raise ValueError('the parameter "hidden_size" must be even!')
		
		# later, n_input_images could correspond to number of layers...
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		self.hidden_size = hidden_size
		
		# generate learnable modules
		self.u1_t = LinearTranspose(image_shape[0]*image_shape[1],hidden_size,bias=False)
		self.w1 = nn.Linear(hidden_size,hidden_size,bias=True)
		
		self.softRELU = nn.Softplus()
		
		self.u1_prime = nn.Linear(hidden_size,image_shape[0]*image_shape[1],bias=True)
		self.w1_prime_t = LinearTranspose(hidden_size,hidden_size,bias=False)
		
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
		self.P2 = torch.mm(R,P).cuda()#probably another multiplication by R needed here... (torch.mm(R,P))
		
	def forward(self,x):
		x_t1 = torch.mean(x[:,self.n_input_images-2,:,:,:],dim=1).view(-1,self.image_shape[0]*self.image_shape[1])
		x_t2 = torch.mean(x[:,self.n_input_images-1,:,:,:],dim=1).view(-1,self.image_shape[0]*self.image_shape[1])
		x1 = batch_mm(self.E1,self.u1_t(x_t1))
		x2 = batch_mm(self.E2,self.u1_t(x_t2))
		x = x1*x2
		x = batch_mm(self.P1,x)
		x = self.softRELU(self.w1_prime_t(x))
		x = batch_mm(self.E3,x)
		x3 = batch_mm(self.E1,self.u1_t(x_t2))
		x = x*x3
		x = batch_mm(self.P2,x)
		x_t3 = self.u1_prime(x).view(-1,self.image_shape[0],self.image_shape[1]).unsqueeze(1)*torch.ones(1,3,1,1).cuda()
		return x_t3
