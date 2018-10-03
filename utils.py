import torch.nn as nn
import torch
import numpy as np
import random
from torch.autograd import Variable
from torch.distributions.categorical import Categorical



def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)


def sampleNoise(batchSize, dimNoise, dimCCont, dimCDisc, test = False):
	#credit to https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
	z = Variable(torch.randn(batchSize, dimNoise,1,1))
	cCont = Variable(torch.FloatTensor(batchSize * dimCCont, 1, 1, 1).uniform_(-1.,1.))
	cCont = Variable(torch.FloatTensor(batchSize,dimCCont,1, 1).uniform_(-1.,1.))
	if test == False:
		y = torch.LongTensor(batchSize,1).random_() % dimCDisc
		#cCont = Variable(torch.FloatTensor(batchSize,dimCCont,1, 1).uniform_(-1.,1.))
	else:
		#cc = [[-2,0],[-1,0],[0,0],[1,0],[2,0],[0,-2],[0,0],[0,2]]*8
		#cCont = Variable(torch.FloatTensor(cc)).resize(batchSize,dimCCont, 1, 1)
		y = torch.LongTensor([[i]*8 for i in range(8)]).resize(batchSize,1)
	cDisc = torch.FloatTensor(batchSize, dimCDisc)
	cDisc.zero_()
	cDisc.scatter_(1,y,1)
	cDisc = cDisc.resize(batchSize,dimCDisc,1,1)
	if test:
		print(cCont)
	z = torch.cat((z, cDisc, cCont),1).resize(batchSize, dimNoise + dimCDisc + dimCCont, 1, 1)	
	return z, cCont, cDisc
