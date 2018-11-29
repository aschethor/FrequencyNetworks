"""
PGP architecture with one layer
optional weight sharing
"""

import sys
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')

from CustomLayers.LinearTranspose import LinearTranspose
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import *
import math

class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100],hidden_size=1000,share_U=True,share_V=True,share_W=True):
		super(Model, self).__init__()
		#later, n_input_images could correspond to number of layers...
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		self.hidden_size = hidden_size
		
		self.u1 = nn.Linear(image_shape[0]*image_shape[1],hidden_size,bias=False)
		self.v1 = nn.Linear(image_shape[0]*image_shape[1],hidden_size,bias=False)
		self.w1 = nn.Linear(hidden_size,hidden_size,bias=False)
		
		self.u1_prime = nn.Linear(image_shape[0]*image_shape[1],hidden_size,bias=False)
		self.v1_prime_t = LinearTranspose(hidden_size,image_shape[0]*image_shape[1],bias=False)
		self.w1_prime_t = LinearTranspose(hidden_size,hidden_size,bias=False)
		
		if share_U:
			self.u1_prime.weight = self.u1.weight
		if share_V:
			self.v1_prime_t.weight = self.v1.weight
		if share_W:
			self.w1_prime_t.weight = self.w1.weight
		
	def forward(self,x):
		"""
		first, 3 input channels are converted into single gray-scale channel
		in the end, this gray-scale is extended back into 3 rgb channels (with the same values)
		"""
		
		x_t1 = torch.mean(x[:,self.n_input_images-2,:,:,:],dim=1).view(-1,self.image_shape[0]*self.image_shape[1])
		x_t2 = torch.mean(x[:,self.n_input_images-1,:,:,:],dim=1).view(-1,self.image_shape[0]*self.image_shape[1])
		
		x1 = self.u1(x_t1)
		x2 = self.v1(x_t2)
		x = x1*x2
		x = torch.sigmoid(self.w1(x))
		x = self.w1_prime_t(x)
		x3 = self.u1_prime(x_t2)
		x = x*x3
		x_t3 = self.v1_prime_t(x).view(-1,self.image_shape[0],self.image_shape[1]).unsqueeze(1)*torch.ones(1,3,1,1).cuda()
		return x_t3
		
