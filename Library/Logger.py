import os
import time
import torch
import datetime as dt
import visdom
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings

class Logger():
	
	def __init__(self,name,datetime=None,use_csv=True,use_visdom=False,use_tensorboard=False):
		"""
		Logger logs metrics into visdom and CSV files
		:name: logging name (e.g. model name / dataset name / ...)
		:datetime: date and time of logging start (useful in case of multiple runs). Default: current date and time is picked
		:use_csv: log output to csv files (needed for plotting)
		:use_visdom: log output to visdom
		:use_tensorboard: log output to tensorboard
		"""
		self.name = name
		if datetime:
			self.datetime=datetime
		else:
			self.datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		self.use_csv = use_csv
		if use_csv:
			os.makedirs('Logger/{}/{}/logs'.format(name,self.datetime),exist_ok=True)
			os.makedirs('Logger/{}/{}/plots'.format(name,self.datetime),exist_ok=True)
		
		self.use_visdom = use_visdom
		if use_visdom:
			self.vis = visdom.Visdom(env="{}_{}".format(name,self.datetime))
			self.windows = {}
		
		self.use_tensorboard = use_tensorboard
		if use_tensorboard:
			directory = 'Logger/tensorboard/{} {}'.format(name,self.datetime)
			os.makedirs(directory,exist_ok=True)
			self.writer = SummaryWriter(directory)
	
	
	def log(self,item,value,index):
		"""
		log index value couple for specific item into csv file
		:item: item
		:value: value
		:index: index
		"""
		
		if self.use_csv:
			filename = 'Logger/{}/{}/logs/{}.log'.format(self.name,self.datetime,item)
			
			if os.path.exists(filename):
				append_write = 'a'
			else:
				append_write = 'w'
			
			with open(filename, append_write) as log_file:
				log_file.write("{}, {}\n".format(index,value))
		
		if self.use_visdom:
			if item not in self.windows:
				self.windows[item] = self.vis.line(X=np.zeros(1),Y=np.zeros(1),opts=dict(title=item,xlabel="index",ylabel=item))
			
			self.windows[item] = self.vis.line(X=np.expand_dims(np.asarray(index),0),Y=np.expand_dims(np.asarray(value),0),update="append",win=self.windows[item])
		
		if self.use_tensorboard:
			self.writer.add_scalar(item,value,index)
	
	
	def plot(self,item, log = False, smoothing = 0.025, ylim = None):
		"""
		plot item metrics
		:item: item
		:log: logarithmic scale. Default: False
		:smoothing: smoothing of metric. Default: 0.025
		:ylim: y-axis limits [lower,upper]
		"""
		if self.use_csv:
			plt.figure(1,figsize=(12,6))
			plt.clf()
			plt.title(self.name)
			plt.ylabel(item)
			plt.xlabel('index')
			train_loss = np.loadtxt(open("Logger/{}/{}/logs/{}.log".format(self.name,self.datetime,item), "rb"), delimiter=",")
			if log:
				plt.semilogy(train_loss[:,0],train_loss[:,1],'r')
			else:
				plt.plot(train_loss[:,0],train_loss[:,1],'r')
			train_loss = lowess(train_loss[:,1],train_loss[:,0], is_sorted=True, frac=smoothing, it=0)
			if log:
				plt.semilogy(train_loss[:,0],train_loss[:,1],'b')
			else:
				plt.plot(train_loss[:,0],train_loss[:,1],'b')
			mean = np.mean(train_loss[:,1])
			std = np.std(train_loss[:,1])
			if log:
				plt.savefig('Logger/{}/{}/plots/{}_log.png'.format(self.name,self.datetime,item),dpi=400)
			else:
				if ylim is not None:
					plt.ylim(ylim)
				else:
					plt.ylim([mean-2*std,mean+4*std])
				plt.savefig('Logger/{}/{}/plots/{}.png'.format(self.name,self.datetime,item),dpi=400)
			
		else:
			warnings.warn("set use_csv=True if you want to plot metrics")
	
	def save_state(self,model,optimizer,index="final"):
		"""
		saves state of model and optimizer
		:model: model to save (if list: save multiple models)
		:optimizer: optimizer (if list: save multiple optimizers)
		:index: index of state to save (e.g. specific epoch)
		"""
		os.makedirs('Logger/{}/{}/states'.format(self.name,self.datetime),exist_ok=True)
		path = 'Logger/{}/{}/states/{}.state'.format(self.name,self.datetime,index)
		state = {}
		
		if type(model)is not list:
			model = [model]
		for i,m in enumerate(model):
			state.update({'model{}'.format(i):m.state_dict()})
		
		if type(optimizer) is not list:
			optimizer = [optimizer]
		for i,o in enumerate(optimizer):
			state.update({'optimizer{}'.format(i):o.state_dict()})
		
		torch.save(state, path)
	
	
	def load_state(self,model,optimizer,datetime,continue_datetime=False,index="final"):
		"""
		loads state of model and optimizer
		:model: model to load (if list: load multiple models)
		:optimizer: optimizer to load (if list: load multiple optimizers; if None: don't load)
		:datetime: date and time from run to load
		:continue_datetime: flag whether to continue on this run. Default: False
		:index: index of state to load (e.g. specific epoch)
		"""
		if continue_datetime:
			self.datetime = datetime
		path = 'Logger/{}/{}/states/{}.state'.format(self.name,datetime,index)
		state = torch.load(path)
		
		if type(model) is not list:
			model = [model]
		for i,m in enumerate(model):
			m.load_state_dict(state['model{}'.format(i)])
		
		if optimizer is not None:
			if type(optimizer)is not list:
				optimizer = [optimizer]
			for i,o in enumerate(optimizer):
				o.load_state_dict(state['optimizer{}'.format(i)])


t_start = 0
t_end = 0


def t_step():
    """
    returns delta t from last call of t_step()
    """
    global t_start,t_end
    t_end = time.clock()
    delta_t = t_end-t_start
    t_start = t_end
    return delta_t
