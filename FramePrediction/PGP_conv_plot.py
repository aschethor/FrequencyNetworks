# plot weights of model

dataset_name = "BouncingBalls"
model_name = "PGP_conv"
load_datetime = "2018-11-23 10:30:12"#"2018-11-23 10:30:02"#
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


share_U=True
share_V=True
share_W=True
share_U=False
share_V=False
share_W=False

# load model
model = Model(n_input_images = input_length,image_shape = image_shape,share_U=share_U,share_V=share_V,share_W=share_W).cuda()
loader = Logger("{}_{}_share_u_{}_v_{}_w_{}".format(dataset_name,model_name,share_U,share_V,share_W),use_csv=False)
loader.load_state(model,None,load_datetime,index=load_epoch)

#print("u1: {}".format(model.u1.weight))
#print("u1_prime: {}".format(model.u1_prime.weight))

n_params = sum(p.numel() for p in model.parameters())
print("number of model parameters: {}".format(n_params))
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of trainable model parameters: {}".format(n_params))
print("model.u1.weight.shape = {}".format(model.u1.weight.shape))

os.makedirs('output/{}_{}/epoch_{}'.format(dataset_name,model_name,load_epoch),exist_ok=True)

def normalize(img):
	print("img.shape = {}".format(img.shape))
	img -= img.mean()
	img /= img.std()*10
	print("img.mean() = {}".format(img.mean()))
	print("img.std() = {}".format(img.std()))
	return img+0.5
	

plt.figure(1,figsize=(2.5,2))
plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(20):
	plt.subplot(4,5,i+1)
	plt.imshow(normalize(model.u1.weight[i].cpu().permute(1,2,0)).detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights u1")
plt.savefig('output/{}_{}/epoch_{}/u1_shared_{}.png'.format(dataset_name,model_name,load_epoch,share_U),dpi=600)

plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(20):
	plt.subplot(4,5,i+1)
	plt.imshow(normalize(model.u1_prime.weight[i].cpu().permute(1,2,0)).detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights u1_prime")
plt.savefig('output/{}_{}/epoch_{}/u1_prime_shared_{}.png'.format(dataset_name,model_name,load_epoch,share_U),dpi=600)

plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(20):
	plt.subplot(4,5,i+1)
	plt.imshow(normalize(model.v1.weight[i].cpu().permute(1,2,0)).detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights v1")
plt.savefig('output/{}_{}/epoch_{}/v1_shared_{}.png'.format(dataset_name,model_name,load_epoch,share_V),dpi=600)

plt.clf()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9,wspace=0.05,hspace=0.05)
for i in range(20):
	plt.subplot(4,5,i+1)
	plt.imshow(normalize(model.v1_prime_t.weight[i].cpu().permute(1,2,0)).detach().numpy())
	plt.axis('off')
	plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.suptitle("weights v1_prime_t")
plt.savefig('output/{}_{}/epoch_{}/v1_prime_t_shared_{}.png'.format(dataset_name,model_name,load_epoch,share_V),dpi=600)
