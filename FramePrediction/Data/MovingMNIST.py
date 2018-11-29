
import torch
import torch.utils.data as data
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

class MovingMNISTDataset(data.Dataset):
	"""BouncingBallsDataset creates image sequences of bouncing balls with random position / velocity"""

	def __init__(self, n_samples, sequence_length, image_shape, digit_velocity = 1, n_digits = 1, digit_color=[torch.FloatTensor([1,1,1])], padding = 0, train = True):
		"""
		:n_samples: size of generated dataset
		:sequence_length: number of frames in sequence
		:image_shape: size of output images (height x width)
		:digit_velocity: velocity of digits
		:n_digits: number of digits
		:ball_color: color of balls
		:padding: border to leave free at the edges
		:train: digits from MNIST training / testing dataset
		"""
		self.n_samples = n_samples
		self.image_shape = image_shape
		self.sequence_length = sequence_length
		self.digit_velocity = digit_velocity
		self.n_digits = n_digits
		self.digit_color = []
		for i in range(n_digits):
			if len(digit_color)==1:
				self.digit_color.append(digit_color[0].unsqueeze(1).unsqueeze(2).cuda())
			else:
				self.digit_color.append(digit_color[i].unsqueeze(1).unsqueeze(2).cuda())
		self.padding = padding
		
		root = "MNIST_DATA"
		self.dataset = dset.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=True)
		return
		
	def __len__(self):
		return self.n_samples

	def __getitem__(self, index):
		"""
		returns:
		:image: image sequence of shape (sequence_length x 3 x height x width)
		"""
		image = torch.zeros(self.sequence_length,3,self.image_shape[0],self.image_shape[1]).cuda()
		positions_x = torch.Tensor(np.random.randint(self.padding,self.image_shape[1]-self.padding-28,self.n_digits)).unsqueeze(1).cuda()
		positions_y = torch.Tensor(np.random.randint(self.padding,self.image_shape[0]-self.padding-28,self.n_digits)).unsqueeze(1).cuda()
		positions = torch.cat([positions_x,positions_y],dim=1)
		angles = np.random.uniform(0,2*np.pi,self.n_digits)
		velocities_x = torch.Tensor(self.digit_velocity*np.cos(angles)).unsqueeze(1).cuda()
		velocities_y = torch.Tensor(self.digit_velocity*np.sin(angles)).unsqueeze(1).cuda()
		velocities = torch.cat([velocities_x,velocities_y],dim=1)
		digit_indices = np.random.randint(0,len(self.dataset),self.n_digits)
		digits = []
		for n in range(self.n_digits):
			digit = self.dataset.__getitem__(digit_indices[n])[0].cuda()
			non_zeros = torch.nonzero(digit)
			min_1 = torch.min(non_zeros[:,1])
			min_2 = torch.min(non_zeros[:,2])
			max_1 = torch.max(non_zeros[:,1])
			max_2 = torch.max(non_zeros[:,2])
			digit = digit[:,min_1:max_1,min_2:max_2]
			digit = digit* self.digit_color[n]
			digits.append(digit)
		
		for i in range(self.sequence_length):
			for n in range(self.n_digits):
				
				x = positions[n,0]
				y = positions[n,1]
				xm = int(positions[n,0])
				ym = int(positions[n,1])
				xp = int(positions[n,0])+1
				yp = int(positions[n,1])+1
				image[i,:,xm:(xm+digits[n].shape[1]),ym:(ym+digits[n].shape[2])] += digits[n]*(xp-x)*(yp-y)
				image[i,:,xp:(xp+digits[n].shape[1]),ym:(ym+digits[n].shape[2])] += digits[n]*(x-xm)*(yp-y)
				image[i,:,xm:(xm+digits[n].shape[1]),yp:(yp+digits[n].shape[2])] += digits[n]*(xp-x)*(y-ym)
				image[i,:,xp:(xp+digits[n].shape[1]),yp:(yp+digits[n].shape[2])] += digits[n]*(x-xm)*(y-ym)
				positions[n,:] = positions[n,:]+velocities[n,:]
				
				# check boundary collisions
				if positions[n,0]<self.padding:
					positions[n,0] += 2*(self.padding-positions[n,0])
					velocities[n,0] *= -1
					
				if positions[n,0]>(self.image_shape[1]-self.padding-digits[n].shape[1]):
					positions[n,0] += 2*(self.image_shape[1]-self.padding-digits[n].shape[1]-positions[n,0])
					velocities[n,0] *= -1
					
				if positions[n,1]<self.padding:
					positions[n,1] += 2*(self.padding-positions[n,1])
					velocities[n,1] *= -1
					
				if positions[n,1]>(self.image_shape[0]-self.padding-digits[n].shape[2]):
					positions[n,1] += 2*(self.image_shape[0]-self.padding-digits[n].shape[2]-positions[n,1])
					velocities[n,1] *= -1
		
		image[image>1] = 1
		return image
