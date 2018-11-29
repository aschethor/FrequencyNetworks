import torch
import torch.nn.functional as F

def trafo_matrices(size):
	"""
	generate util matrices for complex computations
	:size: matrix sizes (should be even)
	returns:
	:I: identity matrix [size x size]
	:P: "pooling" matrix [size x 2*size]
	:B: block matrix
	:R: reordering matrix
	"""
	I = torch.zeros(size,size)
	P = torch.zeros(size,2*size)
	for i in range(size):
		I[i,i]=1
		P[i,2*i]=1
		P[i,2*i+1]=1
	B = torch.zeros(size,size)
	R = torch.zeros(size,size)
	for i in range(1,size,2):
		B[i-1,i]=1
		B[i,i-1]=-1
		R[i-1,int(i/2)]=1
		R[i,int(size/2+i/2)]=-1
	return I,P,B,R

def batch_mm(a,b):
	"""
	matrix multiplication on batch of vectors
	:a: transformation matrix [n x m]
	:b: batch of vectors [s x m]
	:return: transformation performed on b [s x n]
	"""
	return F.linear(b,a,None)
 
def channel_batch_mm(a,b):
	"""
	matrix multiplication on channels of a batch of images
	:a: transformation matrix [n x m]
	:b: batch of images with m channels [s x m x w x h]
	:return: transformation performed on channels [s x n x w x h]
	"""
	shape = b.shape
	b = b.view(shape[0],shape[1],shape[2]*shape[3])
	b = torch.matmul(a,b)
	return b.view(shape[0],a.shape[0],shape[2],shape[3])
