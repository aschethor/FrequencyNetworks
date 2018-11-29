# list of datasets:
# "BouncingBalls"
# "MovingMNIST"
dataset_name = "BouncingBalls"

# list of models:
# "Baseline_1" : baseline (2 convolutional layers with relu activation and batch normalisation)
# "Hourglass_1" : hourglass (3 layers and shortcut connections)
# "Hourglass_2" : hourglass (3 layers and shortcut connections - 2 x more channels as hourglass_1)
# "PGP_1" : Convolutional PGP (1 layer, tiny 5x5 kernels for u,v, relatively big 5x5 kernel for w)
# "PGP_fc" : one layer PGP with fully connected U,V,W (as proposed by Memisevic)
# "PGP_conv" : one layer PGP with convolutional U,V,W
# "PGP_tied_fc" : one layer PGP with tied and fully connected input weights U,W
# "PGP_tied_conv" : one layer PGP with tied and convolutional input weights U,W
# "PGP_Niloofar" : PGP model by Niloofar
# "PGP_tied_Niloofar" : PGP model with tied input weights by Niloofar
model_name = "PGP_tied_conv"
load_datetime = "2018-11-23 14:20:52"
load_epoch = 25
batch_size = 10
input_length = 2#3
prediction_length = 15#30
image_shape = [64,64]#[100,100]
velocity = 1
n = 2#6

import sys
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')

import torch
import matplotlib
matplotlib.use("agg")
matplotlib.rcParams["agg.path.chunksize"]=10000
import matplotlib.pyplot as plt
import Data.BouncingBalls as BB
import Data.MovingMNIST as MM
import os
from Logger import Logger
import numpy as np
Model = __import__(model_name).Model


share_U=False
share_V=False
share_W=False
share_U=True
share_V=True
share_W=True

# load model
if model_name in ["PGP_fc","PGP_conv"]:
	model = Model(n_input_images = input_length,image_shape = image_shape,share_U=share_U,share_V=share_V,share_W=share_W).cuda()
	loader = Logger("{}_{}_share_u_{}_v_{}_w_{}".format(dataset_name,model_name,share_U,share_V,share_W),use_csv=False)
	loader.load_state(model,None,load_datetime,index=load_epoch)
elif model_name in ["PGP_tied_fc","PGP_tied_conv"]:
	model = Model(n_input_images = input_length,image_shape = image_shape,share_U=share_U,share_W=share_W).cuda()
	loader = Logger("{}_{}_share_u_{}_w_{}".format(dataset_name,model_name,share_U,share_W),use_csv=False)
	loader.load_state(model,None,load_datetime,index=load_epoch)
else:
	model = Model(n_input_images = input_length,image_shape = image_shape).cuda()
	loader = Logger("{}_{}".format(dataset_name,model_name),use_csv=False)
	loader.load_state(model,None,load_datetime,index=load_epoch)

#print("u1: {}".format(model.u1.weight))
#print("u1_prime: {}".format(model.u1_prime.weight))

n_params = sum(p.numel() for p in model.parameters())
print("number of model parameters: {}".format(n_params))
unique_params = list(set([p for p in model.parameters()]))
n_params = sum(p.numel() for p in unique_params if p.requires_grad)
#n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of trainable model parameters: {}".format(n_params))
#exit()

# load dataset
if dataset_name == "BouncingBalls":
	ball_radius = 5
	dataset = BB.BouncingBallsDataset(n_samples=batch_size, sequence_length=input_length+prediction_length, image_shape=image_shape, ball_radius=ball_radius, ball_velocity = velocity, n_balls = n, padding = ball_radius)
elif dataset_name == "MovingMNIST":
	dataset = MM.MovingMNISTDataset(n_samples=batch_size, sequence_length=input_length+prediction_length, image_shape=image_shape, digit_velocity = velocity, n_digits = n)

data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

os.makedirs('output/{}_{}/epoch_{}'.format(dataset_name,model_name,load_epoch),exist_ok=True)

plt.figure(1,figsize=(2,10))
plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.05,hspace=0.05)
for batch in data_loader:
	
	# plot initial input frames
	for t in range(input_length):
		for i in range(batch_size):
			# plot ground truth
			plt.subplot(batch_size,2,i*2+1)
			plt.imshow(1-batch[i,t].permute(1,2,0))
			plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
			
			# use ground truth for prediction of input frames
			plt.subplot(batch_size,2,i*2+2)
			plt.imshow(1-batch[i,t].permute(1,2,0))
			plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
			
		plt.savefig('output/{}_{}/epoch_{}/{}.png'.format(dataset_name,model_name,load_epoch,t),dpi=600)
	
	# plot predicted frames
	prediction_batch = batch[:,0:input_length,:,:]
	for t in range(prediction_length):
		prediction = model(prediction_batch)
		prediction_batch = torch.cat([prediction_batch[:,1:input_length,:,:],prediction.unsqueeze(1)],dim=1)
		for i in range(batch_size):
			# plot ground truth
			plt.subplot(batch_size,2,i*2+1)
			plt.imshow(1-batch[i,input_length+t].permute(1,2,0))
			plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
			
			# plot prediction
			plt.subplot(batch_size,2,i*2+2)
			plt.imshow(1-prediction[i].permute(1,2,0).detach().cpu().numpy())
			plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
			
		plt.savefig('output/{}_{}/epoch_{}/{}.png'.format(dataset_name,model_name,load_epoch,input_length+t),dpi=600)
		
	break
