import torch
from torch.autograd import Variable
from torch.nn import init
import math


def gaussian_noise(ins, is_training):
	if is_training:
		noise = Variable(ins.data.new(ins.size()).normal_(mean=0, std=0.1))
		return ins + noise
	return ins


class RAE(torch.nn.Module):
	def __init__(self, N_MAPPINGS, NB_FACTORS, NB_PX):
		super(RAE, self).__init__()
		self.w1 = torch.nn.Parameter(init.xavier_normal(torch.randn(N_MAPPINGS, NB_FACTORS)))
		self.w2 = torch.nn.Parameter(init.xavier_normal(torch.randn(N_MAPPINGS, NB_FACTORS)))
		self.b_mappings = torch.nn.Parameter(init.xavier_normal(torch.randn(1, N_MAPPINGS)))
		self.x_bias = torch.nn.Parameter(init.xavier_normal(torch.randn(1, NB_FACTORS)))
		self.y_bias = torch.nn.Parameter(init.xavier_normal(torch.randn(1, NB_FACTORS)))
		self.m_bias = torch.nn.Parameter(init.xavier_normal(torch.randn(1, NB_FACTORS)))

		self.b_output = torch.nn.Parameter(init.xavier_normal(torch.randn(NB_PX, 1)))
		self.u_x = torch.nn.Parameter(init.orthogonal(torch.randn(NB_PX, NB_FACTORS)))
		self.u_y = torch.nn.Parameter(init.orthogonal(torch.randn(NB_PX, NB_FACTORS)))
		self.m = None
		self.m_prim = None
		self.nonlin = torch.nn.Sigmoid()

	def forward(self, x, y):
		if x is None:
			print("Programming Error")

		self.cor_x1 = gaussian_noise(x, is_training=y is not None)
		self.f_cor_x = torch.addmm(self.x_bias, self.cor_x1, self.u_x)
		device_x = self.f_cor_x.get_device()

		if(self.m is not None):
			self.m_prim = self.m.clone()
		if(y is not None):
			self.cor_y1 = gaussian_noise(y, is_training=True)  # y
			self.f_cor_y = torch.addmm(self.y_bias, self.cor_y1, self.u_y)
			mult_x_y = self.f_cor_x * self.f_cor_y
			self.m = torch.addmm(self.b_mappings, mult_x_y, self.w1.t())
			self.m = self.nonlin(self.m)

		# reconstruction
		recunstructed_factor = torch.addmm(self.m_bias, self.m, self.w2)
		rec_e3 = recunstructed_factor * self.f_cor_x
		self.rec = torch.addmm(self.b_output, self.u_y, rec_e3.t())
		self.rec = self.nonlin(self.rec)
		return self.rec.t()


class Model(torch.nn.Module):
	upper_bound = 15
	count = 0

	def __init__(self, n_input_images,image_shape=[100,100],hidden_size=1000):
		super(Model, self).__init__()
		self.n_input_images = n_input_images
		self.image_shape = image_shape
		self.hidden_size = hidden_size
		
		NB_PX = []
		NB_FACTORS = []
		N_MAPPINGS = []
		
		if n_input_images == 2:
			NB_PX = [image_shape[0]*image_shape[1]]
			NB_FACTORS = [hidden_size]
			N_MAPPINGS = [hidden_size]
		
		self.nb_layers = len(N_MAPPINGS)
		self.mDim = [int(math.sqrt(i)) for i in N_MAPPINGS]
		print(self.mDim, N_MAPPINGS, NB_FACTORS, NB_PX)
		self.rae_layers = torch.nn.ModuleList([RAE(N_MAPPINGS[i], NB_FACTORS[i], NB_PX[i]) for i in range(self.nb_layers)])
		self.div_value = 500
		self.image_m = None
		print("PGP")

	def forward(self, batch):
		# batch :  20, 100, 4096
		batch = torch.mean(batch[:,:,:,:,:],dim=2).view(-1,self.n_input_images,self.image_shape[0]*self.image_shape[1])
		batch = batch.permute(1,0,2)
		#batch = torch.cat([batch,torch.zeros(1,batch.shape[1],batch.shape[2]).cuda()],dim=0)
		
		first_layer_frames = []
		# batch _ size = 40/2
		
		#print("self.n_input_images = {}".format(self.n_input_images))

		first_layer_frames.append(batch[0])
		for i in range(self.n_input_images+1):

			max_available_layer = 1#min(i, self.nb_layers)
			if i < self.n_input_images:  # reconstruct
				for j in range(max_available_layer):
					if j == 0:
						first_layer_frames.append(self.rae_layers[j](batch[i - 1], batch[i]))
					else:
						self.rae_layers[j](self.rae_layers[j - 1].m_prim, self.rae_layers[j - 1].m)  # fill m in self.rae_layers[j]

			elif i < Model.upper_bound:  # predict

				for j in range(max_available_layer - 1, -1, -1):
					if j == 0:
						first_layer_frames.append(self.rae_layers[j](first_layer_frames[-1], None))
					else:
						self.rae_layers[j - 1].m = self.rae_layers[j](self.rae_layers[j - 1].m, None)
			else:

				first_layer_frames.append(batch[i])

		Model.count = Model.count + 1
		#first_layer_frames = torch.stack(first_layer_frames)
		output = first_layer_frames[-1]
		output = output.permute(1,0).contiguous().view(-1,self.image_shape[0],self.image_shape[1]).unsqueeze(1)*torch.ones(1,3,1,1).cuda()
		return output

