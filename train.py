
import sys; sys.path.append("..")
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from timm.scheduler import CosineLRScheduler

from cnn import CNN

# ./sam/example/utility -> ./utility forked from https://github.com/davda54/sam
from utility.log import Log 
from utility.initialize import initialize
from sam import SAM #forked from https://github.com/davda54/sam
from utility.bypass_bn import enable_running_stats, disable_running_stats #forked from https://github.com/davda54/sam

import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

import socket
import datetime

import pkg_resources#save modules
from PIL import Image
import shutil
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

class CXR8CSV_Dataset(torch.utils.data.Dataset):

	def __init__(self, dir, csv_file):
		self.dir = dir
		self.tags = pd.read_csv(csv_file) #number	path	label	position	age	gender
		del self.tags['number']
		self.tags.label		= LabelEncoder().fit_transform(self.tags.label)
		self.tags.position	= LabelEncoder().fit_transform(self.tags.position)

		self.classes = list(OrderedDict.fromkeys(self.tags.label))
		self.targets = self.tags.label
		self.views = self.tags.position
		self.imgs = self.tags

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		width  = round(self.tags.OriginalImage_Width[idx]  * self.tags.OriginalImagePixelSpacing_x[idx])
		height = round(self.tags.OriginalImage_Height[idx] * self.tags.OriginalImagePixelSpacing_y[idx])
		image = Image.open(os.path.join(self.dir,self.tags.path[idx]))\
			.convert('L')\
			.resize((width, height), Image.BICUBIC) # 1.0mm per pixel grayscale image
		label  = self.targets[idx]
		return image, label

class Augmentation(object):
	def __init__(self, perspective=True, rotation=10, pixelsize=[320,320]):
		in_channels=1
		transform = []
		transform.append(transforms.Grayscale(num_output_channels=in_channels))
		transform.append(transforms.RandomPerspective(distortion_scale=0.25, p=0.1, interpolation=transforms.InterpolationMode.BICUBIC, fill=0)) if perspective else ''
		transform.append(transforms.RandomRotation(degrees=rotation)) if rotation > 0 else ''
		transform.append(transforms.CenterCrop(pixelsize))
		transform.append(transforms.ToTensor())
		self.transform = transforms.Compose(transform)

	def __call__(self, img):
		return self.transform(img)

class Subset(torch.utils.data.Dataset):
	def __init__(self, dataset, indices, transform=None):
		self.dataset = dataset
		self.indices = indices
		self.transform = transform

	def __getitem__(self, idx):
		img, label = self.dataset[self.indices[idx]]
		if self.transform:
			img = self.transform(img)
		return img, label

	def __len__(self):
		return len(self.indices)

if __name__ == "__main__":
	print(socket.gethostname())

	parser = argparse.ArgumentParser()
	parser.add_argument('--name', default=None,	help='model name: save folder')
	parser.add_argument('--label_smoothing', default=0.1, type=float, help='Use 0.0 for no label smoothing.') #smooth cross entropy
	parser.add_argument('--rho', default=2.0, type=float, help='Rho parameter for SAM.')
	parser.add_argument('--momentum', default=0.8, type=float, help='SGD Momentum.')
	parser.add_argument('--weight_decay', default=5e-4, type=float, help='L2 weight decay.')
	parser.add_argument('--base_root', default='./')
	parser.add_argument('--model_dir', default='models')
	parser.add_argument('--image_root', default=r'Y:\Studies\DML\Chest\Dataset\CXR8\images')
	parser.add_argument('--dataset_csv', default=r'.\images\CXR8.csv')
	parser.add_argument('--pixelsize', default=[320,320], type=int)
	parser.add_argument('--arch', default='tf_efficientnetv2_s')
	parser.add_argument('--in_channels', default=1, type=int)
	parser.add_argument('--rotation', default=10, type=int)
	parser.add_argument('--perspective', default=True, type=bool)
	parser.add_argument('--pretrain', default=False, type=bool) 
	parser.add_argument('--num_patients', default=0, type=int, help='number of patients')
	parser.add_argument('-b', '--batch_size', default=20,	type=int,	metavar='N', help='mini-batch size')
	parser.add_argument('--drop_last', default=False,	type=bool) 
	parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
	parser.add_argument('--pin_memory', default=True, type=bool)
	parser.add_argument('--warmup_t', default=0, type=float)
	parser.add_argument('--warmup_lr_init', default=5e-3, type=float)
	parser.add_argument('--lr', default=2e-2, type=float)
	parser.add_argument('--epochs', default=300, type=int, help='Total number of epochs.')
	parser.add_argument('--lr_min', default=1e-4, type=float)
	parser.add_argument('--hide_classes', default=1024, type=int)
	parser.add_argument('--drop_out', default=0.5)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

	args = parser.parse_args()

	if args.name is None:
		args.name = 'dropout-%.0e_ep-%d_px-%d' %(
						args.drop_out, 
						args.epochs, 
						args.pixelsize[0]
						)

	save_dir = '%s\\%s\\%s' %(args.base_root, args.model_dir, args.name)
	print('Savedir exists!') if os.path.exists(save_dir) else os.makedirs(save_dir, exist_ok=True)

	modules_txt		= save_dir + '\\modules.txt'
	args_txt		= save_dir + '\\args.txt'
	log_csv			= save_dir + '\\log.csv'
	currentmodel_cpt= save_dir + '\\currentmodel.cpt'
	bestmodel_cpt 	= save_dir + '\\bestmodel.cpt'
	trainset_csv 	= save_dir + '\\trainset.csv'
	validset_csv 	= save_dir + '\\validset.csv'

	initialize(args, seed=42)

	# import dataset
	data_set = CXR8CSV_Dataset(args.image_root,args.dataset_csv)

	# number of patients
	args.num_patients = len(data_set.classes)
	print(f'num_patients: {args.num_patients}')

	train_idx,valid_idx = [], []
	if os.path.isfile(trainset_csv) and os.path.isfile(validset_csv):
		train_idx = pd.read_csv(trainset_csv).iloc[:, 0] #0: index
		valid_idx = pd.read_csv(validset_csv).iloc[:, 0] #0: index
		print('train/valid index exists!')
	else:
		train_idx, valid_idx = train_test_split(
						list(range(len(data_set.targets))),
						test_size=args.num_patients*2, shuffle=True,
						stratify=data_set.targets
					)
		img_table = pd.DataFrame(
						data_set.imgs,
						columns=['path', 'label'],
					)
		img_table.loc[train_idx].to_csv(trainset_csv)
		img_table.loc[valid_idx].to_csv(validset_csv)

	train_set = Subset(data_set, train_idx, Augmentation(   perspective=args.perspective, rotation=args.rotation,
															pixelsize=args.pixelsize,
															))
	valid_set = Subset(data_set, valid_idx, Augmentation(	perspective=False, rotation=0,
															pixelsize=args.pixelsize,
															))

	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last
		)
	valid_loader = torch.utils.data.DataLoader(
		valid_set, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last
		)

	print('Config -----')
	for arg in vars(args):
		print('%s: %s' %(arg, getattr(args, arg)))
	print('------------')

	with open(args_txt, 'w') as f:
		for arg in vars(args):
			print('%s: %s' %(arg, getattr(args, arg)), file=f)
	#save modules
	with open(modules_txt, 'w') as f:
		for p in pkg_resources.working_set:
			print('%s: %s' %(p.project_name, p.version), file=f)

	model = CNN(	model_name=f'{args.arch}',  
					pretrained=args.pretrain,
					px_size=args.pixelsize,
					n_classes=args.num_patients,
					hide_classes=args.hide_classes, 
					drop_out=args.drop_out
					).to(args.device, non_blocking=True)

	criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)

	optimizer = SAM(	model.parameters(), 
						torch.optim.SGD, 
						rho=args.rho, 
						adaptive=True, 
						lr=args.lr, 
						momentum=args.momentum, 
						weight_decay=args.weight_decay
						)

	scheduler = CosineLRScheduler(	optimizer, 
									t_initial=args.epochs, 
									lr_min=args.lr_min, 
									warmup_t=args.warmup_t,
									warmup_lr_init=args.warmup_lr_init, 
									warmup_prefix=True
									)

	elapse_log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'train_loss', 'train_acc1', 'valid_loss', 'valid_acc1', 'time', 'best'])
	elapse_log.to_csv(log_csv, mode='w', index=False) if not (os.path.isfile(log_csv)) else ''

	initial_epoch = 0
	if (os.path.isfile(currentmodel_cpt)):
		checkpoint = torch.load(currentmodel_cpt, torch.device(args.device))
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])

		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(args.device, non_blocking=True)
		initial_epoch = checkpoint['epoch'] + 1

	log = Log(	log_each=10, 
				initial_epoch=initial_epoch
				)

	#read best acc1
	best_acc1 = checkpoint['bestacc1'] if (os.path.isfile(currentmodel_cpt) and 'bestacc1' in checkpoint) else 0.	
	print(f'initial epoch: {initial_epoch}')
	print('best validation acc1: %.4f' %(best_acc1))

	torch.backends.cudnn.benchmark = True

	for epoch in range(initial_epoch, args.epochs):
		model.train()
		log.train(len_dataset=len(train_loader))

		for batch in train_loader:
			inputs, targets = (b.to(args.device, non_blocking=True) for b in batch)

			# first forward-backward step
			enable_running_stats(model)
			outputs = model(inputs, targets)
			loss = criterion(outputs, targets)
			loss.mean().backward()
			optimizer.first_step(zero_grad=True)

			# second forward-backward step
			disable_running_stats(model)
			criterion(model(inputs, targets), targets).mean().backward()
			optimizer.second_step(zero_grad=True)

			with torch.no_grad():
				correct = torch.argmax(outputs.data, 1) == targets
				log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]["lr"])
				scheduler.step(epoch)

		train_log = pd.DataFrame([
			epoch,
			log.learning_rate,
			log.epoch_state['loss']/log.epoch_state['steps'],
			log.epoch_state['accuracy']/log.epoch_state['steps'],
		], columns=[epoch], index=['epoch', 'lr', 'train_loss', 'train_acc1']).T

		model.eval()
		log.eval(len_dataset=len(valid_loader))

		with torch.no_grad():
			for batch in valid_loader:
				inputs, targets = (b.to(args.device, non_blocking=True) for b in batch)

				outputs = model(inputs, targets)
				loss = criterion(outputs, targets)
				correct = torch.argmax(outputs, 1) == targets
				log(model, loss.cpu(), correct.cpu())

		valid_log = pd.DataFrame([
			log.epoch_state['loss']/log.epoch_state['steps'],
			log.epoch_state['accuracy']/log.epoch_state['steps'],
			datetime.datetime.now(),
		], columns=[epoch], index=['valid_loss', 'valid_acc1', 'time']).T

		#checkpoint
		torch.save({'epoch': epoch,
					'model': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict(),
					'loss': log.epoch_state['loss'],
					'accuracy': log.epoch_state['accuracy'],
					'steps': log.epoch_state['steps'],
					'bestacc1': valid_log.at[epoch, 'valid_acc1'] if best_acc1 < valid_log.at[epoch, 'valid_acc1'] else best_acc1,
					},
				currentmodel_cpt)

		#Save Best model
		best_log = pd.DataFrame([False],index=[epoch], columns=['best'])
		if best_acc1 < valid_log.at[epoch, 'valid_acc1']:
			shutil.copyfile(currentmodel_cpt, bestmodel_cpt)
		pd.concat([train_log, valid_log, best_log], axis=1).to_csv(log_csv, mode='a', index=False, header=False)

	log.flush()
