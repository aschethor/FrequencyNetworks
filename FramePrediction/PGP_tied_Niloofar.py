import torch
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import scipy.linalg
import math

def get_block_diagonal_matrix(nb_blocs, bloc):
	if nb_blocs <= 0:
		return np.array([])

	mat = bloc
	for i in range(nb_blocs-1):
		mat = scipy.linalg.block_diag(mat, bloc)
	return np.array(mat)

def get_pooling_matrix(L):
	return get_block_diagonal_matrix(L, [1,1])

def get_block_diagonal_matrix_b(L):
	b = get_block_diagonal_matrix(math.floor(L / 2), [[0,1],[-1,0]])
	if L > 0 and L % 2 != 0:
		b = scipy.linalg.block_diag(b, [0])
	return b

def get_reordering_matrix(L):
	if L <= 0:
		return np.array([])
	elif L == 1:
		return np.array([1])

	r_part1 = get_block_diagonal_matrix(math.floor(L / 2), [[1],[0]])
	r_part2 = get_block_diagonal_matrix(math.floor(L / 2), [[0],[-1]])

	r = []
	if L % 2 != 0:
		r = np.concatenate((r_part1, np.zeros((L-1,1))), axis=1)
		r = np.concatenate((r, r_part2), axis=1)
		ligne_f = np.zeros((1,L))
		ligne_f[0][math.floor(L/2)] = 1.
		r = np.concatenate((r, ligne_f), axis=0)
	else:
		r = np.concatenate((r_part1, r_part2), axis=1)

	return r 


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
		self.m = None
		self.m_prim = None
		self.nonlin = torch.nn.Sigmoid()
		self.nonlinRelu = torch.nn.ReLU()
		self.u = torch.nn.Parameter(init.orthogonal(torch.randn(NB_PX, NB_FACTORS)))
		self.p = Variable(torch.from_numpy(get_pooling_matrix(NB_FACTORS).astype(np.float32)),  requires_grad= False).cuda()
		self.b = torch.from_numpy(get_block_diagonal_matrix_b(NB_FACTORS).astype(np.float32))
		self.r = torch.from_numpy(get_reordering_matrix(NB_FACTORS).astype(np.float32))
		inp1 = init.eye(torch.Tensor(NB_FACTORS,NB_FACTORS))
		self.e1 = Variable(torch.cat((inp1,  inp1), 0), requires_grad= False).cuda()
		self.e2 = torch.cat((inp1, self.b), 0)
		self.e2 = Variable(self.e2, requires_grad= False).cuda()
		self.e3 = torch.cat((self.r, torch.mm(self.b, self.r)), 0)
		self.e3 = Variable(self.e3, requires_grad= False).cuda()


	#Formula 10 except that: my input is in the form of: batch*NB_PX thus we need to transpose them as well during multiplication. 
	def forward(self, x, y):
		if x is None:
			print("Programming Error")
		self.cor_x1 = gaussian_noise(x, is_training=y is not None)
		#The next two lines: e1u^Tx^T in m in Formula 10
		self.f_cor_x = torch.addmm(self.x_bias, self.cor_x1, self.u)
		midle_cor_x = torch.mm(self.e1, self.f_cor_x.t())
		device_x = midle_cor_x.get_device()

		if(self.m is not None):
			self.m_prim = self.m.clone()
		if(y is not None):
			self.cor_y1 = gaussian_noise(y, is_training=True)  # y
			#The next two lines: e2u^Ty^T in m in Formula 10
			self.f_cor_y = torch.addmm(self.y_bias, self.cor_y1, self.u)
			midle_cor_y = torch.mm(self.e2, self.f_cor_y.t())
			#Elementwise multiplication(*) in m in Formula 10
			mult_x_y = midle_cor_x * midle_cor_y
			mult_x_y  = torch.mm(self.p, mult_x_y) 
			self.m = torch.addmm(self.m_bias, mult_x_y.t(), self.w1)
			self.m = self.nonlin(self.m)

		# reconstruction (r in Formula 11)
		reconstructed_factor = torch.mm(self.m, self.w2)
		recunstructed_factor = torch.mm(self.e3, reconstructed_factor.t())
		rec_e3 = recunstructed_factor * midle_cor_x
		recunstructed_frame = torch.mm(self.u, torch.mm(self.p, rec_e3))
		self.rec = recunstructed_frame.add_(self.b_output.expand_as(recunstructed_frame) )
		
		return self.nonlin(self.rec.t())


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
		batch = torch.cat([batch,torch.zeros(1,batch.shape[1],batch.shape[2]).cuda()],dim=0)
		
		first_layer_frames = []
		# batch _ size = 40/2

		first_layer_frames.append(batch[0])
		for i in range(len(batch)):

			max_available_layer = min(i, self.nb_layers)
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
		first_layer_frames = torch.stack(first_layer_frames)
		output = first_layer_frames[len(first_layer_frames)-1]
		output = output.permute(1,0).contiguous().view(-1,self.image_shape[0],self.image_shape[1]).unsqueeze(1)*torch.ones(1,3,1,1).cuda()
		return output

