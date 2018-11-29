import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100]):
		super(Model, self).__init__()
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		
		self.conv1 = nn.Conv2d( n_input_images*3, 20, kernel_size=5,padding = 2)
		self.bn1 = nn.BatchNorm2d(20)
		self.conv2 = nn.Conv2d( 20, 3, kernel_size=5,padding=2)
		
	def forward(self,x):
		x = x.view(-1,self.n_input_images*3,self.image_shape[0],self.image_shape[1])
		x = torch.relu(self.bn1(self.conv1(x)))
		x = torch.relu(self.conv2(x))
		return x
