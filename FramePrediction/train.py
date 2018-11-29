# list of datasets:
# "BouncingBalls"
# "MovingMNIST"
dataset_name = "MovingMNIST"

# list of models:
# "Baseline_1" : baseline (2 convolutional layers with relu activation and batch normalisation)
# "Hourglass_1" : hourglass (3 layers and shortcut connections)
# "Hourglass_2" : hourglass (3 layers and shortcut connections - 2 x more channels as hourglass_1)
# "PGP_1" : Convolutional PGP (1 layer, tiny 5x5 kernels for u,v, relatively big 5x5 kernel for w)
# "PGP_fc" : one layer PGP with fully connected U,V,W (as proposed by Memisevic)
# "PGP_conv" : one layer PGP with convolutional U,V,W
# "PGP_conv_2" : one layer PGP with convolutional U,V,W (bigger kernels for U/V, smaller kernels but more channels for W, works on grayscale images)
# "PGP_tied_fc" : one layer PGP with tied and fully connected input weights U,W
# "PGP_tied_fc_2" : one layer PGP with tied and fully connected input weights U,W with softrelu, and changed P2... -> performed worse
# "PGP_tied_fc_3" : one layer PGP with tied and fully connected input weights U,W with softrelu -> worse than PGP_tied_fc
# "PGP_tied_conv" : one layer PGP with tied and convolutional input weights U,W
# "PGP_tied_conv_2" : one layer PGP with tied and convolutional input weights U,W (bigger kernels for U, smaller kernels but more channels for W, works on grayscale images)
# "PGP_tied_conv_3" : one layer PGP with tied and convolutional input weights U,W (bigger kernels for U, smaller kernels but more channels for W, works on grayscale images) with softrelu, and changed P2... -> didn't work
# "PGP_tied_conv_4" : one layer PGP with tied and convolutional input weights U,W (bigger kernels for U, smaller kernels but more channels for W, works on grayscale images) with softrelu -> worse than PGP_tied_conv_2
# "PGP_Niloofar" : PGP model by Niloofar
# "PGP_tied_Niloofar" : PGP model with tied input weights by Niloofar
model_name = "PGP_tied_fc"

batch_size = 40
n_epochs = 100
n_batches_per_epoch = 500
input_length = 2 # number of input frames for the model (3 needed at least to predict interactions)
prediction_length = 5 # number of output frames during training (model predicts frames 1 by 1)
image_shape = [64,64] #image_shape = [100,100] #"2018-11-14 15:58:13"
velocity = 1 # initial ball/digit velocity (play with that!)
n = 1#2#6 # more balls -> interactions should be learned more efficiently
use_tensorboard = True
use_csv = True

share_U=False
share_V=False
share_W=False
share_U=True
share_V=True
share_W=True

import sys
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')

import torch
import Data.BouncingBalls as BB
import Data.MovingMNIST as MM
Model = __import__(model_name).Model
import torch.optim as optim
from Logger import Logger
import numpy as np

print("start training of {}_n_{}_{}_share_u_{}_v_{}_w_{}".format(dataset_name,n,model_name,share_U,share_V,share_W))

# load model and initialize logger
if model_name in ["PGP_fc","PGP_conv","PGP_conv_2"]:
	model = Model(n_input_images = input_length,image_shape = image_shape,share_U=share_U,share_V=share_V,share_W=share_W).cuda()
	logger = Logger("{}_n_{}_{}_share_u_{}_v_{}_w_{}".format(dataset_name,n,model_name,share_U,share_V,share_W),use_csv=use_csv,use_tensorboard=use_tensorboard)
elif model_name in ["PGP_tied_fc","PGP_tied_fc_2","PGP_tied_fc_3","PGP_tied_conv","PGP_tied_conv_2","PGP_tied_conv_3","PGP_tied_conv_4"]:
	model = Model(n_input_images = input_length,image_shape = image_shape,share_U=share_U,share_W=share_W).cuda()
	logger = Logger("{}_n_{}_{}_share_u_{}_w_{}".format(dataset_name,n,model_name,share_U,share_W),use_csv=use_csv,use_tensorboard=use_tensorboard)
else:
	model = Model(n_input_images = input_length,image_shape = image_shape).cuda()
	logger = Logger("{}_n_{}_{}".format(dataset_name,n,model_name),use_csv=use_csv,use_tensorboard=use_tensorboard)
Loss = torch.nn.modules.loss.MSELoss()

# load dataset
if dataset_name == "BouncingBalls":
	ball_radius = 5
	dataset = BB.BouncingBallsDataset(n_samples=batch_size*n_batches_per_epoch, sequence_length=input_length+prediction_length, image_shape=image_shape, ball_radius=ball_radius, ball_velocity = velocity, n_balls = n, padding = ball_radius)
elif dataset_name == "MovingMNIST":
	dataset = MM.MovingMNISTDataset(n_samples=batch_size*n_batches_per_epoch, sequence_length=input_length+prediction_length, image_shape=image_shape, digit_velocity = velocity, n_digits = n)

data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

# load optimizer and define loss
optimizer = optim.Adam(model.parameters(),lr=3e-5)#TODO:try 5e-5
Loss = torch.nn.modules.loss.MSELoss()

# start training
for epoch in range(n_epochs):
	for i,batch in enumerate(data_loader):
		input_frames = batch[:,0:(input_length),:,:]
		loss = 0
		for t in range(prediction_length):
			prediction = model(input_frames)
			loss += Loss(prediction,batch[:,input_length+t,:,:,:]) # evtl add weights for different timesteps...
			input_frames = torch.cat([input_frames[:,1:input_length,:,:,:],prediction.unsqueeze(1)],dim=1)
		loss /= prediction_length
		
		print("batch {}/{}: loss = {}".format(i,len(data_loader),loss))
		logger.log("loss",loss.detach().cpu().numpy(),epoch*n_batches_per_epoch+i)
		
		optimizer.zero_grad()
		loss.backward()
		clip = 2
		torch.nn.utils.clip_grad_norm(model.parameters(),clip)
		optimizer.step()
		#print("model.u1_t.weight==model.u1_prime.weight : {}".format(model.u1_t.weight==model.u1_prime.weight))
		
		if (i+1)%20 == 0:
			logger.plot("loss")
			logger.plot("loss",log=True)
		
	logger.save_state(model,optimizer,epoch+1)
