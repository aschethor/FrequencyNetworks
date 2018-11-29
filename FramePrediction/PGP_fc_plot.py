# plot weights of model

dataset_name = "MovingMNIST"#"BouncingBalls"
n=1
model_name = "PGP_fc"
load_datetime = "2018-11-28 09:36:07"#"2018-11-23 10:29:46"#"2018-11-23 10:29:37"#
load_epoch = 25
input_length = 2
batch_size = 10
image_shape = [64,64]#[100,100]

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
model = Model(n_input_images = input_length,image_shape = image_shape,share_U=share_U,share_V=share_V,share_W=share_W).cuda()
loader = Logger("{}_n_{}_{}_share_u_{}_v_{}_w_{}".format(dataset_name,n,model_name,share_U,share_V,share_W),use_csv=False)
loader.load_state(model,None,load_datetime,index=load_epoch)

#print("u1: {}".format(model.u1.weight))
#print("u1_prime: {}".format(model.u1_prime.weight))

n_params = sum(p.numel() for p in model.parameters())
print("number of model parameters: {}".format(n_params))
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of trainable model parameters: {}".format(n_params))


os.makedirs('output/{}_n_{}_{}/epoch_{}'.format(dataset_name,n,model_name,load_epoch),exist_ok=True)

plt.figure(1,figsize=(4,2))
plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(200):
	plt.subplot(10,20,i+1)
	plt.imshow(model.u1.weight[i].view(image_shape[0],image_shape[1]).cpu().detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights u1")
plt.savefig('output/{}_n_{}_{}/epoch_{}/u1_shared_{}.png'.format(dataset_name,n,model_name,load_epoch,share_U),dpi=600)

plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(200):
	plt.subplot(10,20,i+1)
	plt.imshow(model.u1_prime.weight[i].view(image_shape[0],image_shape[1]).cpu().detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights u1_prime")
plt.savefig('output/{}_n_{}_{}/epoch_{}/u1_prime_shared_{}.png'.format(dataset_name,n,model_name,load_epoch,share_U),dpi=600)

plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(200):
	plt.subplot(10,20,i+1)
	plt.imshow(model.v1.weight[i].view(image_shape[0],image_shape[1]).cpu().detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights v1")
plt.savefig('output/{}_n_{}_{}/epoch_{}/v1_shared_{}.png'.format(dataset_name,n,model_name,load_epoch,share_V),dpi=600)

plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(200):
	plt.subplot(10,20,i+1)
	plt.imshow(model.v1_prime_t.weight[i].view(image_shape[0],image_shape[1]).cpu().detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights v1_prime_t")
plt.savefig('output/{}_n_{}_{}/epoch_{}/v1_prime_t_shared_{}.png'.format(dataset_name,n,model_name,load_epoch,share_V),dpi=600)