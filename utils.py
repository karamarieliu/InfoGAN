import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)


def sampleNoise(batchSize, dimNoise, dimCCont, dimCDisc):
	z = Variable(torch.randn(batchSize * dimNoise, 1, 1, 1))

	cCont = Variable(torch.FloatTensor(batchSize * dimCCont, 1, 1, 1).uniform_(-1.,1.))
	prob = 1/dimCDisc
	cat = Categorical(torch.tensor([prob]*dimCDisc))
	n = []
	for i in range(dimCDisc*batchSize):
		n.append(cat.sample())
	cDisc = Variable(torch.FloatTensor(n).resize(dimCDisc*batchSize, 1, 1, 1))
	z = torch.cat((z, cDisc, cCont)).resize(batchSize, dimNoise + dimCDisc + dimCCont, 1, 1)

	return z, cCont, cDisc