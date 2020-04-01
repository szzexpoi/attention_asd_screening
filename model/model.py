import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import os

class G_LSTM(nn.Module):
	"""
	LSTM implementation proposed by A. Graves (2013),
	it has more parameters compared to original LSTM
	"""
	def __init__(self,input_size=2048,hidden_size=512):
		super(G_LSTM,self).__init__()
		# without batch_norm
		self.input_x = nn.Linear(input_size,hidden_size,bias=True)
		self.forget_x = nn.Linear(input_size,hidden_size,bias=True)
		self.output_x = nn.Linear(input_size,hidden_size,bias=True)
		self.memory_x = nn.Linear(input_size,hidden_size,bias=True)

		self.input_h = nn.Linear(hidden_size,hidden_size,bias=True)
		self.forget_h = nn.Linear(hidden_size,hidden_size,bias=True)
		self.output_h = nn.Linear(hidden_size,hidden_size,bias=True)
		self.memory_h = nn.Linear(hidden_size,hidden_size,bias=True)

		self.input_c = nn.Linear(hidden_size,hidden_size,bias=True)
		self.forget_c = nn.Linear(hidden_size,hidden_size,bias=True)
		self.output_c = nn.Linear(hidden_size,hidden_size,bias=True)

	def forward(self,x,state):
		h, c = state
		i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input_c(c))
		f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_c(c))
		g = torch.tanh(self.memory_x(x) + self.memory_h(h))

		next_c = torch.mul(f,c) + torch.mul(i,g)
		o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output_c(next_c))
		h = torch.mul(o,next_c)
		state = (h,next_c)

		return state

class Sal_seq(nn.Module):
	def __init__(self,backend,seq_len,hidden_size=512):
		super(Sal_seq,self).__init__()
		self.seq_len = seq_len
		# defining backend
		if backend == 'resnet':
			resnet = models.resnet50(pretrained=True)
			self.init_resnet(resnet)
			input_size = 2048
		elif backend == 'vgg':
			vgg = models.vgg19(pretrained=True)
			self.init_vgg(vgg)
			input_size = 512
		else:
			assert 0, 'Backend not implemented'

		self.rnn = G_LSTM(input_size,hidden_size)
		self.decoder = nn.Linear(hidden_size,1,bias=True) # comment for multi-modal distillation
		self.hidden_size = hidden_size

	def init_resnet(self,resnet):
		self.backend = nn.Sequential(*list(resnet.children())[:-2])

	def init_vgg(self,vgg):
		# self.backend = vgg.features
		self.backend = nn.Sequential(*list(vgg.features.children())[:-2]) # omitting the last Max Pooling

	def init_hidden(self,batch): #initializing hidden state as all zero
		h = torch.zeros(batch,self.hidden_size).cuda()
		c = torch.zeros(batch,self.hidden_size).cuda()
		return (Variable(h),Variable(c))

	def process_lengths(self,input):
		"""
		Computing the lengths of sentences in current batchs
		"""
		max_length = input.size(1)
		lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
		return lengths

	def crop_seq(self,x,lengths):
		"""
		Adaptively select the hidden state at the end of sentences
		"""
		batch_size = x.size(0)
		seq_length = x.size(1)
		mask = x.data.new().resize_as_(x.data).fill_(0)
		for i in range(batch_size):
			mask[i][lengths[i]-1].fill_(1)
		mask = Variable(mask)
		x = x.mul(mask)
		x = x.sum(1).view(batch_size, x.size(2))
		return x

	def forward(self,x,fixation):
		valid_len = self.process_lengths(fixation) # computing valid fixation lengths
		x = self.backend(x)
		batch, feat, h, w = x.size()
		x = x.view(batch,feat,-1)

		# recurrent loop
		state = self.init_hidden(batch) # initialize hidden state
		fixation = fixation.view(fixation.size(0),1,fixation.size(1))
		fixation = fixation.expand(fixation.size(0),feat,fixation.size(2))
		x = x.gather(2,fixation)
		output = []
		for i in range(self.seq_len):
			# extract features corresponding to current fixation
			cur_x = x[:,:,i].contiguous()
			#LSTM forward
			state = self.rnn(cur_x,state)
			output.append(state[0].view(batch,1,self.hidden_size))

		# selecting hidden states from the valid fixations without padding
		output = torch.cat(output, 1)
		output = self.crop_seq(output,valid_len)
		output = torch.sigmoid(self.decoder(output))
		return output
