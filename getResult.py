import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision
import PIL
from PIL import Image
from PIL import ImageChops
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim

######################################################################
preprocess = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor()
])

postprocess = torchvision.transforms.Compose([
	torchvision.transforms.ToPILImage()
])

testSize=0
batchSize=1

def getRandomPatch(image,w):		#getting random patch of w*w from image
	width=image.size[0]
	height=image.size[1]
	startx=random.randint(0,width-w)
	starty=random.randint(0,height-w)
	box=(startx,starty,startx+w,starty+w)
	return image.crop(box)

def getDiffRes(HR8):			#getting different processed image (bilinear,bicubic interpolated) along with their components
	global preprocess
	width=HR8.size[0]
	height=HR8.size[1]
	HR4=HR8.resize((width/2,height/2))
	HR2=HR4.resize((width/4,height/4))
	LR=HR2.resize((width/8,height/8))
	LRR,LRG,LRB=LR.split()
	HR2R,HR2G,HR2B=HR2.split()
	HR4R,HR4G,HR4B=HR4.split()
	HR8R,HR8G,HR8B=HR8.split()
	BL2=LR.resize((width/4,height/4),Image.BILINEAR)
	BL4=BL2.resize((width/2,height/2),Image.BILINEAR)
	BL8=BL4.resize((width,height),Image.BILINEAR)
	BC2=LR.resize((width/4,height/4),Image.BICUBIC)
	BC4=BL2.resize((width/2,height/2),Image.BICUBIC)
	BC8=BL4.resize((width,height),Image.BICUBIC)
	return [LR,LRR,LRG,LRB,HR2,HR2R,HR2G,HR2B,HR4,HR4R,HR4G,HR4B,HR8,HR8R,HR8G,HR8B,BL2,BL4,BL8,BC2,BC4,BC8]

class imageDataset(Dataset):	#getting processed image dataset
	def __init__(self,root_dir,testData):
		self.root_dir = root_dir
		self.transform = torchvision.transforms.ToTensor()
		self.testData=testData

	def __len__(self):
		return len(self.testData)

	def __getitem__(self, idx):
		image=PIL.Image.open(self.root_dir+'/'+self.testData[idx])
		# image=getRandomPatch(image,64)
		[LR,LRR,LRG,LRB,HR2,HR2R,HR2G,HR2B,HR4,HR4R,HR4G,HR4B,HR8,HR8R,HR8G,HR8B,BL2,BL4,BL8,BC2,BC4,BC8]=getDiffRes(image)
		return {'LR':self.transform(LR),
				'LRR':self.transform(LRR),'LRG':self.transform(LRG),
				'LRB':self.transform(LRB),
				'HR2':self.transform(HR2),
				'HR2R':self.transform(HR2R),'HR2G':self.transform(HR2G),
				'HR2B':self.transform(HR2B),
				'HR4':self.transform(HR4),
				'HR4R':self.transform(HR4R),'HR4G':self.transform(HR4G),
				'HR4B':self.transform(HR4B),
				'HR8':self.transform(HR8),
				'HR8R':self.transform(HR8R),'HR8G':self.transform(HR8G),
				'HR8B':self.transform(HR8B),
				'BL2':self.transform(BL2),'BL4':self.transform(BL4),'BL8':self.transform(BL8),
				'BC2':self.transform(BC2),'BC4':self.transform(BC4),'BC8':self.transform(BC8)
				}

def getTestData():			#getting test data
	global testSize,batchSize
	dirName='test'
	testData=os.listdir(dirName)
	DataSet=imageDataset('test',testData)
	dataloaders =torch.utils.data.DataLoader(DataSet, batch_size=batchSize,
                                             shuffle=False, num_workers=1)
	return dataloaders
######################################################################################################################
class featuresNet(nn.Module):	#defining features branch net

	def __init__(self,firstLevel):
		super(featuresNet, self).__init__()
		self.firstLevel=firstLevel
		if self.firstLevel==True:
			self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
		self.conv2=nn.Conv2d(64, 64, 3, padding=1)
		self.conv3=nn.Conv2d(64, 64, 3, padding=1)
		self.conv4=nn.Conv2d(64, 64, 3, padding=1)
		self.conv5=nn.Conv2d(64, 64, 3, padding=1)
		self.conv6=nn.Conv2d(64, 64, 3, padding=1)
		self.conv7=nn.Conv2d(64, 64, 3, padding=1)
		self.conv8=nn.Conv2d(64, 64, 3, padding=1)
		self.conv9=nn.Conv2d(64, 64, 3, padding=1)
		self.conv10=nn.Conv2d(64, 64, 3, padding=1)
		self.conv11=nn.Conv2d(64, 64, 3, padding=1)

		self.transposedConv = nn.ConvTranspose2d(64,64,4,stride=2, padding=1)
		self.convres = nn.Conv2d(64,1,3,padding=1)

	def forward(self, x):
		if self.firstLevel==True:
			x = F.leaky_relu(self.conv1(x),0.2,False)
		x = F.leaky_relu(self.conv2(x),0.2,False)
		x = F.leaky_relu(self.conv3(x),0.2,False)
		x = F.leaky_relu(self.conv4(x),0.2,False)
		x = F.leaky_relu(self.conv5(x),0.2,False)
		x = F.leaky_relu(self.conv6(x),0.2,False)
		x = F.leaky_relu(self.conv7(x),0.2,False)
		x = F.leaky_relu(self.conv8(x),0.2,False)
		x = F.leaky_relu(self.conv9(x),0.2,False)
		x = F.leaky_relu(self.conv10(x),0.2,False)
		x = F.leaky_relu(self.conv11(x),0.2,False)
		x=self.transposedConv(x)
		y=self.convres(x)

		return [x,y]

class imageReconstructNet(nn.Module):	#defining image reconstruction branch net

	def __init__(self):
		super(imageReconstructNet, self).__init__()
		self.transposedConv = nn.ConvTranspose2d(1,1,4,stride=2, padding=1)

	def forward(self, x):
		x=self.transposedConv(x)
		return x

class LAPSRN(nn.Module):				# defining the full lapsrn
	"""docstring for LAPSRN"""
	def __init__(self):
		super(LAPSRN, self).__init__()
		self.featuresNet1=featuresNet(True)
		self.imgReconstructNet1=imageReconstructNet()
		self.featuresNet2=featuresNet(False)
		self.imgReconstructNet2=imageReconstructNet()
		self.featuresNet3=featuresNet(False)
		self.imgReconstructNet3=imageReconstructNet()
		self.featuresNet1.apply(weights_init)
		self.featuresNet2.apply(weights_init)
		self.featuresNet3.apply(weights_init)
		self.imgReconstructNet1.apply(weights_init1)
		self.imgReconstructNet2.apply(weights_init1)
		self.imgReconstructNet3.apply(weights_init1)

	def forward(self,x):
		resList=[]
		features1=self.featuresNet1(x)
		imgReconstruct1=self.imgReconstructNet1(x)
		imgReconstruct1=imgReconstruct1+features1[1]
		features2=self.featuresNet2(features1[0])
		imgReconstruct2=self.imgReconstructNet2(imgReconstruct1)
		imgReconstruct2=imgReconstruct2+features2[1]
		features3=self.featuresNet3(features2[0])
		imgReconstruct3=self.imgReconstructNet3(imgReconstruct2)
		imgReconstruct3=imgReconstruct3+features3[1]
		return [imgReconstruct1,imgReconstruct2,imgReconstruct3]

def weights_init(m): 	# initialize features net kernel weights 
	classname = m.__class__.__name__
	if classname.find('ConvTranspose2d') != -1:
		a=np.array([[0.0625,0.1875,0.1875,0.0625],[ 0.1875,0.5625,0.5625,0.1875],[ 0.1875 , 0.5625 , 0.5625 , 0.1875],[ 0.0625 , 0.1875 , 0.1875 , 0.0625]])
		b=np.zeros((64,64,4,4))
		for i in xrange(64):
			for j in xrange(64):
				b[i][j]=a
		c=torch.Tensor(b)
		m.weight.data.copy_(c)

def weights_init1(m):	# initialize image reconstruction net kernel weights 
	classname = m.__class__.__name__
	if classname.find('ConvTranspose2d') != -1:
		a=np.array([[0.0625,0.1875,0.1875,0.0625],[ 0.1875,0.5625,0.5625,0.1875],[ 0.1875 , 0.5625 , 0.5625 , 0.1875],[ 0.0625 , 0.1875 , 0.1875 , 0.0625]])
		b=np.zeros((1,1,4,4))
		for i in xrange(1):
			for j in xrange(1):
				b[i][j]=a
		c=torch.Tensor(b)
		m.weight.data.copy_(c)

def CharbonierLoss(A,B):	# calculating charbonier loss function for training
	x=B-A
	x=x*x
	epsilon=1e-3
	y = Variable(torch.Tensor([epsilon]).float())
	y=y.cuda()
	y=y*y
	z = x + y.expand(x.size())
	z=z.sqrt()
	return z.sum()

def convert(tensor):	# converting tensor to cuda variable
	res=Variable(tensor)
	res=res.cuda()
	return res

def rmseLoss(A,B):		# calculating mean square error
	x=B-A
	x=x*x
	z= x.size()
	return x.sum()/(z[0]*z[1]*z[2]*z[3])

test_data=getTestData()
lapnet=torch.load('mytraining.pt')	#loading trained model

try:
	os.mkdir('result')				#making folders for results
except OSError:
	pass
i=0
for batch in test_data:				#applying model on each image
	LR=[convert(batch['LRR']),convert(batch['LRG']),convert(batch['LRB'])]
	outR = lapnet(LR[0])
	outG = lapnet(LR[1])
	outB = lapnet(LR[2])
	BL2=postprocess(torch.squeeze(batch['BL2']))
	BL4=postprocess(torch.squeeze(batch['BL4']))
	BL8=postprocess(torch.squeeze(batch['BL8']))
	BC2=postprocess(torch.squeeze(batch['BC2']))
	BC4=postprocess(torch.squeeze(batch['BC4']))
	BC8=postprocess(torch.squeeze(batch['BC8']))
	HR2=postprocess(torch.squeeze(batch['HR2']))
	HR4=postprocess(torch.squeeze(batch['HR4']))
	HR8=postprocess(torch.squeeze(batch['HR8']))
	LRini=postprocess(torch.squeeze(batch['LR']))
	HR2_target=[outR[0].cpu().data,outG[0].cpu().data,outB[0].cpu().data]
	HR4_target=[outR[1].cpu().data,outG[1].cpu().data,outB[1].cpu().data]
	HR8_target=[outR[2].cpu().data,outG[2].cpu().data,outB[2].cpu().data]
	HR2_target=postprocess(torch.squeeze(torch.cat((HR2_target[0],HR2_target[1],HR2_target[2]),1)))
	HR4_target=postprocess(torch.squeeze(torch.cat((HR4_target[0],HR4_target[1],HR4_target[2]),1)))
	HR8_target=postprocess(torch.squeeze(torch.cat((HR8_target[0],HR8_target[1],HR8_target[2]),1)))
	filename='result/'+str(i)
	try:
		os.mkdir(filename)
	except OSError:
		pass
	BC2.save(filename+'/BicubicX2.jpg')
	BC4.save(filename+'/BicubicX4.jpg')
	BC8.save(filename+'/BicubicX8.jpg')
	HR2.save(filename+'/RealX2.jpg')
	HR4.save(filename+'/RealX4.jpg')
	HR8.save(filename+'/RealX8.jpg')
	HR2_target.save(filename+'/ResultX2.jpg')
	HR4_target.save(filename+'/ResultX4.jpg')
	HR8_target.save(filename+'/ResultX8.jpg')
	LRini.save(filename+'/LR.jpg')
	i+=1
	print 'image '+str(i)+' done'