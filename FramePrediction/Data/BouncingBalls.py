
import torch
import torch.utils.data as data
import numpy as np

class BouncingBallsDataset(data.Dataset):
	"""BouncingBallsDataset creates image sequences of bouncing balls with random position / velocity"""

	def __init__(self, n_samples, sequence_length, image_shape, ball_radius, ball_velocity = 1, n_balls = 1, ball_color=[torch.FloatTensor([1,1,1])], padding = 0):
		"""
		:n_samples: size of generated dataset
		:sequence_length: number of frames in sequence
		:image_shape: size of output images (height x width)
		:ball_radius: radius of balls
		:ball_velocity: velocity of balls
		:n_balls: number of balls
		:ball_color: color of balls
		:padding: border to leave free at the edges
		"""
		self.n_samples = n_samples
		self.image_shape = image_shape
		self.sequence_length = sequence_length
		self.ball_radius = ball_radius
		self.ball_velocity = ball_velocity
		self.n_balls = n_balls
		self.ball_color = []
		for i in range(n_balls):
			if len(ball_color)==1:
				self.ball_color.append(ball_color[0].unsqueeze(1).unsqueeze(2).cuda())
			else:
				self.ball_color.append(ball_color[i].unsqueeze(1).unsqueeze(2).cuda())
		self.padding = padding
		self.grid_y, self.grid_x = torch.meshgrid([torch.linspace(1,self.image_shape[0],self.image_shape[0]),torch.linspace(1,self.image_shape[1],self.image_shape[1])])
		self.grid_x = self.grid_x.cuda()
		self.grid_y = self.grid_y.cuda()
		self.ones = torch.ones([3,1,1]).cuda()
		return
		
	def __len__(self):
		return self.n_samples

	def ball(self, position):
		return (torch.exp(-((self.grid_x-position[0])**2+(self.grid_y-position[1])**2)/self.ball_radius**2).unsqueeze(0)*self.ones*np.e)**5

	def __getitem__(self, index):
		"""
		returns:
		:image: image sequence of shape (sequence_length x 3 x height x width)
		"""
		warmup = 10
		image = torch.zeros(self.sequence_length+warmup,3,self.image_shape[0],self.image_shape[1]).cuda()
		positions_x = torch.Tensor(np.random.randint(self.ball_radius+self.padding,self.image_shape[1]-self.ball_radius-self.padding,self.n_balls)).unsqueeze(1).cuda()
		positions_y = torch.Tensor(np.random.randint(self.ball_radius+self.padding,self.image_shape[0]-self.ball_radius-self.padding,self.n_balls)).unsqueeze(1).cuda()
		positions = torch.cat([positions_x,positions_y],dim=1)
		angles = np.random.uniform(0,2*np.pi,self.n_balls)
		velocities_x = torch.Tensor(self.ball_velocity*np.cos(angles)).unsqueeze(1).cuda()
		velocities_y = torch.Tensor(self.ball_velocity*np.sin(angles)).unsqueeze(1).cuda()
		velocities = torch.cat([velocities_x,velocities_y],dim=1)
		for i in range(self.sequence_length+warmup):
			for n in range(self.n_balls):
				image[i] = image[i] + self.ball(positions[n,:])*self.ball_color[n]
				positions[n,:] = positions[n,:]+velocities[n,:]
				
				# check ball collisions
				for m in range(self.n_balls):
					if m!=n and torch.norm(positions[n,:]-positions[m,:])<2*self.ball_radius:
						# lazy method to reposition ball m
						for k in range(-3,5):
							if torch.norm(positions[n,:]-positions[m,:])<2*self.ball_radius:
								positions[n,:] -= velocities[n,:]*0.5**k
							else:
								positions[n,:] += velocities[n,:]*0.5**k
						positions[n,:] -= velocities[n,:]*0.5**4
						unit_distance = positions[m,:]-positions[n,:]
						unit_distance /= torch.norm(unit_distance)
						n_proj = torch.sum(unit_distance*velocities[n,:])*unit_distance
						m_proj = torch.sum(unit_distance*velocities[m,:])*unit_distance
						mean_velocity = (n_proj+m_proj)/2
						velocities[n,:] += 2*(mean_velocity-n_proj)
						velocities[m,:] += 2*(mean_velocity-m_proj)
				
				# check boundary collisions
				if positions[n,0]<self.padding:
					positions[n,0] += 2*(self.padding-positions[n,0])
					velocities[n,0] *= -1
					
				if positions[n,0]>(self.image_shape[1]-self.padding):
					positions[n,0] += 2*(self.image_shape[1]-self.padding-positions[n,0])
					velocities[n,0] *= -1
					
				if positions[n,1]<self.padding:
					positions[n,1] += 2*(self.padding-positions[n,1])
					velocities[n,1] *= -1
					
				if positions[n,1]>(self.image_shape[0]-self.padding):
					positions[n,1] += 2*(self.image_shape[0]-self.padding-positions[n,1])
					velocities[n,1] *= -1
		
		image[image>1] = 1
		#image = torch.sigmoid(image**3*1000)
		return image[warmup:(self.sequence_length+warmup)]
