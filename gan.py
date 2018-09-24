import torch
import torch.nn as nn
import torchvision.datasets as dset
import utils 

class Gen(nn.Module):
	def __init__(self, c, numFilters, dimNoise, imgSize, dimLatent): 
		super(Gen, self).__init__()

		self.dimNoise = dimNoise + dimLatent
		self.numFiltersG = numFilters
		self.model = nn.Sequential(
			nn.Linear(1,1, bias = False),
			nn.ConvTranspose2d(self.dimNoise, numFilters * 4, 4, 1, 0, bias=False),
			nn.BatchNorm2d(numFilters * 4, momentum = .8),
			nn.LeakyReLU(.2, inplace=True),
			
			nn.ConvTranspose2d(numFilters * 4, numFilters * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters * 2, momentum = .8),
			nn.LeakyReLU(.2, inplace=True),

			nn.ConvTranspose2d(numFilters * 2, numFilters, 2, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters, momentum = .8),
			nn.LeakyReLU(.2, inplace=True),

			nn.ConvTranspose2d(    numFilters,      c, 4, 2, 1, bias=False),
			nn.Linear(imgSize,imgSize,bias=False),
			nn.Tanh()
			)			
		utils.initialize_weights(self)

	def forward(self, z):
		return self.model(z)


class DisFront(nn.Module):
	#Computes D(G(z,c)) up to the point where Q and D diverge. See _Back() for final completion. 
	def __init__(self, c, numFilters, imgSize): 
		super(DisFront, self).__init__()
		self.numFiltersD = numFilters

		self.model = nn.Sequential(
			nn.Linear(imgSize,imgSize, bias=False),
			nn.Conv2d(c, numFilters, 4, 2, 1, bias=False),
			nn.LeakyReLU(.2, inplace=True),

			nn.Conv2d(numFilters, numFilters * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters * 2, momentum = .2),
			nn.LeakyReLU(.2, inplace=True),

			nn.Conv2d(numFilters * 2, numFilters * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(numFilters * 4, momentum = .2),
			nn.LeakyReLU(.2, inplace=True),
			)
		utils.initialize_weights(self)

	def forward(self, img):
		x = self.model(img)
		return x

class DisBack(nn.Module):
	#Computes the back half of D(G(z,c)). 
	def __init__(self, numFilters): 
		super(DisBack, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(numFilters * 4,     1, 4, 2, 1, bias=False),
			nn.Linear(1,1, bias=False),
			nn.Sigmoid()
			)		
		utils.initialize_weights(self)
	
	def forward(self, img):
		x= self.model(img)
		return x.view(-1,1)

class QBack(nn.Module):
	#Computes the back half of Q(c | x). 
	def __init__(self, numFilters): 
		super(QBack, self).__init__()		
		self.model = nn.Sequential(
			nn.Conv2d(numFilters * 4,     10, 4, 2, 1, bias=False),
			nn.BatchNorm2d(10, momentum = .2),
			nn.LeakyReLU(.2, inplace=True),
			nn.Linear(1,1, bias=False),
			)				
		utils.initialize_weights(self)


	def forward(self, img):
		x = self.model(img)
		return x.squeeze()



