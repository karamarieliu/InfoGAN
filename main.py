import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torch.optim as op
import torchvision.transforms as transforms
import torchvision.utils as utilsT 
from torch.autograd import Variable
import math
from torch.distributions.categorical import Categorical
import gan
import numpy as np
import random 
import utils
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, default = 'mnist', help='only mnist supported so far')
parser.add_argument('--dataRoot', default='./mnist', help='where our training and test dataset will be')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='(height, width) of the input image to network')
parser.add_argument('--dimNoise', type=int, default=62, help='size of the latent z vector, or the noise as input to gen')
parser.add_argument('--dimCDisc', type=int, default=10, help='size of the latent c, as a discrete variable')
parser.add_argument('--dimCCont', type=int, default=2, help='size of the latent c, as a continuous variable')
parser.add_argument('--numFiltersG', type=int, default=128)
parser.add_argument('--numFiltersD', type=int, default=128)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0001')
parser.add_argument('--c', type=int, default=1, help='numChannels - unsure')
parser.add_argument('--b1', type=float, default=.5, help='beta1 for Adam')
parser.add_argument('--b2', type=float, default=.999, help='beta2 for Adam')
parser.add_argument('--saveInterval', default=1000, help='interval number we save after')
parser.add_argument('--imageRoot', default= './images', help='images sampled from test data')
parser.add_argument('--lam', type=float, default= 1, help='mutual information weight')
parser.add_argument('--restore', type=int, default= 0, help='0 if we are starting a new model. 1 if we want to restore a previous model')



hp = parser.parse_args()
print(hp)


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

if hp.dataset == 'mnist':
	if not os.path.exists(hp.dataRoot):
			os.mkdir(hp.dataRoot)

	trans = transforms.Compose([
							   transforms.Resize(hp.imageSize),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_set = dset.MNIST(root=hp.dataRoot, train=True, transform=trans, download=True)
	test_set = dset.MNIST(root=hp.dataRoot, train=False, transform=trans, download=True)

	train_loader = torch.utils.data.DataLoader(
				 dataset=train_set,
				 batch_size=hp.batchSize,
				 shuffle=True)
	test_loader = torch.utils.data.DataLoader(
				dataset=test_set,
				batch_size=hp.batchSize,
				shuffle=False)
print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
print ('==>>> total testing batch number: {}'.format(len(test_loader)))

if not os.path.exists(hp.imageRoot):
	os.mkdir(hp.imageRoot)

gen = gan.Gen(hp.c, hp.numFiltersG, hp.dimNoise, hp.imageSize, hp.dimCDisc+hp.dimCCont)
disF = gan.DisFront(hp.c, hp.numFiltersD, hp.imageSize)
disB = gan.DisBack(hp.numFiltersD)
qB = gan.QBack(hp.numFiltersD)


gOpt = op.Adam(gen.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))
dOpt = op.Adam(disF.parameters(), lr=hp.lr,betas=(hp.b1, hp.b2))


lossF_D = nn.BCELoss() #loss function for Discriminator

for epoch in range(hp.niter):

	for itr, (imgsReal, _) in enumerate(train_loader):

				z, cCont, cDisc = utils.sampleNoise(hp.batchSize, hp.dimNoise, hp.dimCCont, hp.dimCDisc)

				#Optimization for Discriminator. 
				dOpt.zero_grad()
	  		
				actualDReal= disF(imgsReal)
				actualDReal = disB(actualDReal)
				expectedDReal = Variable(torch.FloatTensor(actualDReal.size()[0],1).fill_(random.uniform(.9, 1.0)), requires_grad=False)
				lossReal = lossF_D(actualDReal, expectedDReal)


				imgGFake = gen(z)
				actualDFake = disF(imgGFake)
				actualDFake = disB(actualDFake)
				expectedDFake = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.0, 0.1)), requires_grad=False)
				lossFake = lossF_D(actualDFake, expectedDFake)

				dLoss = (lossFake + lossReal) * .5
				dLoss.backward()
				dOpt.step()

				#Optimization for Generator.
				gOpt.zero_grad()
				imgGFake2 = gen(z)

				actualGFake1 = disF(imgGFake2)
				actualGFake = disB(actualGFake1)
				expectedGFake = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.9, 1.0)), requires_grad=False)
				gLossTrick = lossF_D(actualGFake, expectedGFake)

				#Mutual information loss.

				#Discrete variable loss
				qResult = qB(actualGFake1)
				minAct = cDisc.squeeze()
				mInfLossDis = torch.mean(torch.softmax(qResult, dim=0).reshape(hp.batchSize * 10) * minAct)

				#Continuous variable loss 
				var = torch.diag(torch.var(qResult, dim=1))

				varI = torch.inverse(var)
				c = cCont.squeeze().reshape(hp.batchSize, -1)
				mu = torch.mean(qResult, dim=1)
				mInfLossCon = 0
				for i in range(hp.batchSize):
					mInfLossCon += -math.log(var[i][i]) - .5*varI[i][i]*(pow((c[i][0]-mu[i]),2) + pow((c[i][1]-mu[i]),2))
				mInfLossCon = mInfLossCon/hp.batchSize
				qLoss = -hp.lam*(mInfLossCon + mInfLossDis)

				gLoss = gLossTrick + qLoss
				gLoss.backward()
				gOpt.step()
				
		
				if (itr % 25) == 0:
					print("Epoch: [%2d] Iter: [%2d] D_loss: %.8f, G_loss: %.8f" % (epoch, itr, dLoss.item(), gLoss.item()))
						


				if (itr % hp.saveInterval) == 0:
					torch.save(disF.state_dict(), 'savedModels/disF_%03d_%03d.ckpt' % (epoch, itr) )
					torch.save(disB.state_dict(), 'savedModels/disB_%03d_%03d.ckpt' % (epoch, itr) ) 
					torch.save(qB.state_dict(),'savedModels/qB_%03d_%03d.ckpt' % (epoch, itr) )
					torch.save(gen.state_dict(), 'savedModels/gen_%03d_%03d.ckpt' % (epoch, itr) ) 
					z2, cCont2, cDisc2 = utils.sampleNoise(hp.batchSize, hp.dimNoise, hp.dimCCont, hp.dimCDisc)

					#Testing the Generator.
					imgFakeTest = gen(z2)
					actualFakeTest = disF(imgFakeTest)
					actualFakeTest = disB(actualFakeTest)
					expectedFakeTestG = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.9, 1.0)), requires_grad=False)
					lossFakeTestG = lossF_D(actualFakeTest, expectedFakeTestG)

					#Testing the Discriminator. 
					expectedFakeTestD = Variable(torch.FloatTensor(hp.batchSize,1).fill_(random.uniform(0.,.1)), requires_grad=False)
					lossFakeTestD = lossF_D(actualFakeTest, expectedFakeTestD)

					for itr2, (imgsRealTest, _) in enumerate(test_loader):
						if itr2 < 1: 
							actualRealTest= disF(imgsRealTest)
							actualRealTest= disB(actualRealTest)
							expectedRealTest = Variable(torch.FloatTensor(actualRealTest.size()[0],1).fill_(random.uniform(.9,1.0)), requires_grad=False)
							lossRealTest = lossF_D(actualRealTest, expectedRealTest)
							print("Testing D loss: [%2d]  Testing G loss: [%2d]" % (.5 * (lossRealTest + lossFakeTestD), lossFakeTestG))
							utilsT.save_image(imgFakeTest.detach(), 'images/testSampleFake_%03d_%03d.png' % (epoch, itr), normalize=True)
