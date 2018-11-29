"""
Hourglass-architecture
"""

import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
	
	def __init__(self,n_input_images,image_shape=[100,100]):
		super(Model, self).__init__()
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		
		self.conv1 = nn.Conv2d( n_input_images*3, 40, kernel_size=5,stride = 2,padding = 2)
		self.conv2 = nn.Conv2d( 40, 40, kernel_size=5,stride = 2,padding = 0)
		self.conv3 = nn.Conv2d( 40, 40, kernel_size=5,stride = 2,padding = 0)
		self.deconv1 = nn.ConvTranspose2d( 40, 40, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 80, 40, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 80, 40, kernel_size=2, stride = 2, padding=0)
		
		self.conv4 = nn.Conv2d( 40+n_input_images*3, 3, kernel_size=5,padding = 2)
		
	def forward(self,x):
		x1 = x.view(-1,self.n_input_images*3,self.image_shape[0],self.image_shape[1])
		x2 = torch.relu(self.conv1(x1))
		x3 = torch.relu(self.conv2(x2))
		x = torch.relu(self.conv3(x3))
		x = torch.relu(self.deconv1(x))
		x = torch.cat([x,x3],dim=1)
		x = torch.relu(self.deconv2(x, output_size = [50,50]))
		x = torch.cat([x,x2],dim=1)
		x = torch.relu(self.deconv3(x, output_size = self.image_shape))
		x = torch.cat([x,x1],dim=1)
		x = torch.relu(self.conv4(x))
		return x
