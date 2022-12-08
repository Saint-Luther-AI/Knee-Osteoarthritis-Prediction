import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv


class Solver(object):
	def __init__(self, config, train_loader):

		# Data loader
		self.train_loader = train_loader


		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=4)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img

	def get_accuracy(SR, GT):
		SR = F.softmax(SR, dim=1)
		SR = torch.argmax(SR, dim=1).squeeze(1)
		# GT = torch.argmax(GT, dim=1).squeeze(1)
		corr = torch.sum(SR == GT)
		tensor_size = SR.size(0) * SR.size(1) * SR.size(2)
		acc = float(corr) / float(tensor_size)
		return acc


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		#loss function
		#weights = torch.tensor([0.1, 0.8, 0.1], dtype=torch.float)
		#weights = weights.to(self.device)
		#self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
		self.criterion = torch.nn.CrossEntropyLoss()
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):

					# GT : Ground Truth
					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = self.unet(images)
					SR = SR.to(self.device)
					GT = torch.argmax(GT, dim=1).squeeze(1)
					loss = self.criterion(SR, GT)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

				best_unet = self.unet.state_dict()
				torch.save(best_unet,unet_path)
				print(epoch, epoch_loss)

