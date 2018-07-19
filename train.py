import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim

#######################################################################################################################

preprocess = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor()
])

postprocess = torchvision.transforms.Compose([
	torchvision.transforms.ToPILImage()
])

batchSize=20
trainSetIncrease=1
lossList=[]
epochs=1

def getRandomPatch(image):		#getting random patch of 128*128from image
	width=image.size[0]
	height=image.size[1]
	startx=random.randint(0,width-128)
	starty=random.randint(0,height-128)
	box=(startx,starty,startx+128,starty+128)
	return image.crop(box)

def getDiffRes(HR8):			#returns different processed image (bilinear,bicubic interpolated) along with their components
	global preprocess
	HR4=HR8.resize((64,64))
	HR2=HR4.resize((32,32))
	LR=HR2.resize((16,16))
	LRR,LRG,LRB=LR.split()
	HR2R,HR2G,HR2B=HR2.split()
	HR4R,HR4G,HR4B=HR4.split()
	HR8R,HR8G,HR8B=HR8.split()
	return [LR,LRR,LRG,LRB,HR2,HR2R,HR2G,HR2B,HR4,HR4R,HR4G,HR4B,HR8,HR8R,HR8G,HR8B]

def alterRandomImage(image):	# returns a completely randomly rotated and scaled image
	width=image.size[0]
	height=image.size[1]
	
	scale=random.uniform(0.5,1)
	retImg=image.resize((int(width*scale),int(height*scale)),Image.BICUBIC)

	fliplr=random.randint(0,1)
	flipud=random.randint(0,1)
	if fliplr==1:
		retImg=retImg.transpose(Image.FLIP_LEFT_RIGHT)
	if flipud==1:
		retImg=retImg.transpose(Image.FLIP_TOP_BOTTOM)

	rotate=random.randint(0,3)
	if rotate==1:
		retImg=retImg.transpose(Image.ROTATE_90)
	elif rotate==2:
		retImg=retImg.transpose(Image.ROTATE_180)
	elif rotate==3:
		retImg=retImg.transpose(Image.ROTATE_270)

	return retImg

class imageDataset(Dataset):	#getting processed image dataset
	def __init__(self,root_dir,trainData):
		self.root_dir = root_dir
		self.transform = torchvision.transforms.ToTensor()
		self.trainData=trainData

	def __len__(self):
		return len(self.trainData)

	def __getitem__(self, idx):
		image=PIL.Image.open(self.root_dir+'/'+self.trainData[idx])
		image=alterRandomImage(image)
		imagePatch=getRandomPatch(image)
		[LR,LRR,LRG,LRB,HR2,HR2R,HR2G,HR2B,HR4,HR4R,HR4G,HR4B,HR8,HR8R,HR8G,HR8B]=getDiffRes(imagePatch)
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
				'HR8B':self.transform(HR8B)}

def getTrainData():	#getting train data
	global batchSize
	trainData=os.listdir('train')
	DataList=[]
	for i in xrange(trainSetIncrease):
		DataSet=imageDataset('train',trainData)
		DataList.append(DataSet)
	DataSet=torch.utils.data.ConcatDataset(DataList)
	dataloaders =torch.utils.data.DataLoader(DataSet, batch_size=batchSize,
                                             shuffle=True, num_workers=1)
	return dataloaders

def getValidData():	#getting validation data
	global batchSize
	trainData=os.listdir('val')
	DataList=[]
	for i in xrange(trainSetIncrease):
		DataSet=imageDataset('val',trainData)
		DataList.append(DataSet)
	DataSet=torch.utils.data.ConcatDataset(DataList)
	dataloaders =torch.utils.data.DataLoader(DataSet, batch_size=1,
                                             shuffle=True, num_workers=1)
	return dataloaders

##############################################################################

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

lapnet =LAPSRN()
# lapnet=torch.load('mytraining.pt')
lapnet=lapnet.cuda()
lossList=[]
optimizer = optim.Adam(lapnet.parameters(),lr=10e-5)
file=open('loss.csv','a')		#file to save train loss
file1=open('lossVal.csv','a')	# file to save validation loss
train_data=getTrainData()		# getting training data
valid_data=getValidData()		# getting validation data

print 'Training ......'			# training begins
for i in xrange(epochs):
	print 'epoch'+str(i)+':'
	epochLoss=0
	el=0
	for batch in train_data:
		LR=[convert(batch['LRR']),convert(batch['LRG']),convert(batch['LRB'])]
		HR2=[convert(batch['HR2R']),convert(batch['HR2G']),convert(batch['HR2B'])]
		HR4=[convert(batch['HR4R']),convert(batch['HR4G']),convert(batch['HR4B'])]
		HR8=[convert(batch['HR8R']),convert(batch['HR8G']),convert(batch['HR8B'])]
		optimizer.zero_grad()   # zero the gradient buffers
		outR = lapnet(LR[0])
		outG = lapnet(LR[1])
		outB = lapnet(LR[2])

		HR2_target=[outR[0],outG[0],outB[0]]
		HR4_target=[outR[1],outG[1],outB[1]]
		HR8_target=[outR[2],outG[2],outB[2]]

		loss=0
		loss1=0
		for j in xrange(3):
			loss+=CharbonierLoss(HR2[j],HR2_target[j])
			loss1+=rmseLoss(HR2[j],HR2_target[j])
		for j in xrange(3):
			loss+=CharbonierLoss(HR4[j],HR4_target[j])
			loss1+=rmseLoss(HR4[j],HR4_target[j])
		for j in xrange(3):
			loss+=CharbonierLoss(HR8[j],HR8_target[j])
			loss1+=rmseLoss(HR8[j],HR8_target[j])
		el+=loss.data[0]
		loss.backward()
		epochLoss+=loss1.data[0]
		optimizer.step()    # Does the update
	file.write(str(epochLoss)+','+str(el))
	if i%10==0: 	# validating at some epochs
		print 'Validating for epoch:'+str(i)+'......'
		for batch in valid_data:
			LR=[convert(batch['LRR']),convert(batch['LRG']),convert(batch['LRB'])]
			HR2=[convert(batch['HR2R']),convert(batch['HR2G']),convert(batch['HR2B'])]
			HR4=[convert(batch['HR4R']),convert(batch['HR4G']),convert(batch['HR4B'])]
			HR8=[convert(batch['HR8R']),convert(batch['HR8G']),convert(batch['HR8B'])]
			outR = lapnet(LR[0])
			outG = lapnet(LR[1])
			outB = lapnet(LR[2])
			HR2_target=[outR[0],outG[0],outB[0]]
			HR4_target=[outR[1],outG[1],outB[1]]
			HR8_target=[outR[2],outG[2],outB[2]]
			loss=0
			loss1=0
			for j in xrange(3):
				loss+=CharbonierLoss(HR2[j],HR2_target[j])
				loss1+=rmseLoss(HR2[j],HR2_target[j])
			for j in xrange(3):
				loss+=CharbonierLoss(HR4[j],HR4_target[j])
				loss1+=rmseLoss(HR4[j],HR4_target[j])
			for j in xrange(3):
				loss+=CharbonierLoss(HR8[j],HR8_target[j])
				loss1+=rmseLoss(HR8[j],HR8_target[j])
			el+=loss.data[0]
			epochLoss+=loss1.data[0]
		file1.write(str(epochLoss)+','+str(el)+'\n')
		print 'Done Validation'
	print 'Saving mytraining'+str(i)+'.pt'
	torch.save(lapnet,'mytrainingEpoch'+str(i)+'.pt')
print 'Training Done'
file.close()
file1.close()
torch.save(lapnet,'mytraining.pt')
print 'Saving mytraining.pt'