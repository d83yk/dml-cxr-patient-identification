
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from cnn import CNN

from torchvision import transforms
from pathlib import Path
import warnings

import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn

import shutil#.copyfile as copyfile
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

from sklearn.metrics import roc_curve, auc
import seaborn as sns

class CheXpertCSV_Dataset(Dataset):

	def __init__(self, dir, csv_file, type='baseline'):
		self.tags = pd.read_csv(csv_file) #     number,path,position,folder,label,study,view,Sex,Age,No Finding
		self.numbers = self.tags.number

		self.labels = self.tags.baseline_patient if type=='baseline' else self.tags.followup_patient
		self.paths  = 	[os.path.join(dir, p) for p in self.tags.baseline]\
			if type=='baseline'	else\
						[os.path.join(dir, p) for p in self.tags.followup]
		self.matches    = self.tags.match

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		image = Image.open(self.paths[idx]).convert('L') # from PIL import Image
		label  = self.labels[idx]
		match = self.matches[idx]
		return image, label, match

class Transform(object):
	def __init__(
			self,
			pixelsize=[320,390],
		):
		transform = []
		transform.append(transforms.Grayscale(num_output_channels=1))
		transform.append(transforms.CenterCrop(pixelsize))
		transform.append(transforms.ToTensor())
		self.transform = transforms.Compose(transform)
				
	def __call__(self, img):
		return self.transform(img)

class Subset(Dataset):
	def __init__(self, dataset, indices, transform=None):
		self.dataset = dataset
		self.matches   = dataset.matches
		self.paths   = dataset.paths
		self.indices = indices
		self.transform = transform
		subsets = lambda items, indices: [item for idx,item in enumerate(items) if idx in indices]
		labels = subsets(dataset.labels, indices)
		self.classes = list(OrderedDict.fromkeys(labels))

	def __getitem__(self, idx):
		img, label, match = self.dataset[self.indices[idx]]
		if self.transform:
			img = self.transform(img)
		return img, label, match

	def __len__(self):
		return len(self.indices)

def eer(fpr,tpr,th):
	""" Returns equal error rate (EER) and the corresponding threshold. """
	fnr = 1-tpr
	abs_diffs = np.abs(fpr - fnr)
	min_idx = np.argmin(abs_diffs)
	eer = np.mean((fpr[min_idx], fnr[min_idx]))
	return eer, th[min_idx]



def test_verification(baseline_loader,  followup_loader, model, model_path, model_cpt, store_dir, args):
	
	#特徴数
	score_matrix, target1_matrix, target2_matrix = [], [], []
	with torch.no_grad():

		model.to(args.device, non_blocking=True)
		model.eval()	# switch to evaluate mode

		current_model = torch.load(os.path.join(model_path, model_cpt))
		current_epoch = 0
		if os.path.splitext(model_cpt)[1] == '.pth':
			model.load_state_dict(current_model)
		else:# .cpt
			model.load_state_dict(current_model['model'])
			current_epoch = current_model['epoch']

		feature1_vector_list = list()
		feature2_vector_list = list()
		match_list = list()

		for (input2, target2, match2) in tqdm(followup_loader, total=len(followup_loader), desc=str(current_epoch)+"-"+model_cpt):
			feature2_vector = model.featurier(input2.to(args.device, non_blocking=True))
			feature2_vector_list.append(feature2_vector)
			target2_matrix.extend([ i2 for i2 in target2.tolist() ])
			match_list.extend([ i2 for i2 in match2.tolist() ])

		for (input1, target1, _) in tqdm(baseline_loader, total=len(baseline_loader), desc=str(current_epoch)+"-"+model_cpt):
			feature1_vector = model.featurier(input1.to(args.device, non_blocking=True))
			feature1_vector_list.append(feature1_vector)
			target1_matrix.extend([ i1 for i1 in target1.tolist() ])

		for (f2v, f1v) in zip(feature2_vector_list, feature1_vector_list):
			score_matrix.extend(	F.cosine_similarity(
										f2v, 
										f1v,
										dim=1,
										eps=1e-5
										).cpu().detach().numpy().copy()#.tolist()
									)

	same_patient_score = [score_matrix[i] for i, x in enumerate(match_list) if x==1]
	diff_patient_score = [score_matrix[i] for i, x in enumerate(match_list) if x==0]
	same_diff_score = [*same_patient_score, *diff_patient_score] # list
	same_diff_label = [*[1]*len(same_patient_score),*[0]*len(diff_patient_score)] #list

	roc_curve_png = f'{store_dir}\\roc_curve.png'
	fpr, tpr, th = roc_curve(same_diff_label, same_diff_score)
	auc_value = auc(fpr, tpr)
	eer_value, eer_pos = eer(fpr,tpr,th)
	plt.style.use('default')
	sns.set_palette('gray')
	plt.plot(fpr, tpr, marker='None')
	plt.xlabel('False rejection rate')
	plt.ylabel('Correct acceptance rate')
	plt.title(f'ROC Curve  AUC:{"{:.4f}".format(auc_value)}')
	plt.grid()
	plt.savefig(roc_curve_png)
	plt.clf()
	plt.close()

	#histogram
	histogram_png = f'{store_dir}\\histogram.png'
	sns.set()
	sns.set_style('whitegrid')
	sns.set_palette('Set1')
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.hist(same_patient_score,  bins=100, range=(0, 1), density=True, alpha=0.7)
	ax.hist(diff_patient_score, bins=100, range=(0, 1), density=True, alpha=0.7)
	ax.set_xlabel('Cosine Similarity')
	ax.set_ylabel('Relative Frequency')
	plt.title('Genuine-Impostor histogram EER:%.4f %.4f' %(eer_value, eer_pos))
	#plt.show()
	plt.savefig('%s' %(histogram_png))
	plt.clf()
	plt.close()

	score_tsv		= '%sscore.tsv' %(store_dir)
	proc_tsv		= '%spROC.tsv' %(store_dir)	#pROC.tsv
	roc_curve_tsv	= '%sroc_curve_auc=%3f.tsv' %(store_dir, auc_value)
	performance_tsv	= '%sperformance_auc=%3f_eer=%3f.tsv' %(store_dir, auc_value, eer_value)

	#save
	with open(roc_curve_tsv, 'w') as f_handle:
		np.savetxt(f_handle, [fpr, tpr, th], delimiter='\t', fmt="%s")
	with open(score_tsv, 'w') as f_handle:
		np.savetxt(f_handle, [target1_matrix, target2_matrix, match_list, score_matrix], delimiter='\t', fmt="%s")#.reshape(1,-1)
	#pROC format
	pd.DataFrame({
					'score':  same_diff_score,
					'genuine': same_diff_label,
				}).to_csv(proc_tsv, sep='\t', index=False)

	#performance.tsv
	with open(performance_tsv, 'w') as f_handle:
		np.savetxt(f_handle, [['pth', 'AUC', 'EER', 'EER_Pos','Genuine', 'Impostor']], delimiter='\t', fmt="%s")
		np.savetxt(f_handle,np.r_[[model_cpt, auc_value, eer_value, eer_pos, np.average(score_matrix[match_list==1]), np.average(score_matrix[match_list==0])]].reshape(1,-1), delimiter='\t', fmt="%s")
	print('%d-%s: AUC %.4f EER %.4f' %(current_epoch, model_cpt, auc_value, eer_value))


if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--name', default='CheXpert',	help='model name: save folder')
	parser.add_argument('--main_root', default='./')
	parser.add_argument('--model_dir', default='models/dropout-5e-01_ep-300_px-320')
	parser.add_argument('--image_root', default=r'.\images\CheXpert\Small')
	parser.add_argument('--dataset_csv', default=r'.\images\CheXpert.csv')
	parser.add_argument('--pixelsize', default=[320,390], type=int)#CheXpert

	parser.add_argument('--model_root', default=None)
	parser.add_argument('--model_cpt', default='bestmodel.cpt')
	parser.add_argument('--arch', default='tf_efficientnetv2_s')
	parser.add_argument('--train_patients', default=3245, type=int, help='patient num. on training')
	parser.add_argument('--baseline_patients', default=0, type=int, help='patient num. verification')
	parser.add_argument('--followup_patients', default=0, type=int, help='patient num. verification')
	parser.add_argument('--hide_classes', default=1024, type=int, help='dimention of hidden features')
	parser.add_argument('-b', '--batch_size', default=10,	type=int,	metavar='N', help='mini-batch size') 
	parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
	parser.add_argument('--pin_memory', default=True, type=bool)
	parser.add_argument('--drop_last', default=False,	type=bool)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args()

	warnings.simplefilter('ignore')

	store = Path(f'{os.path.join(args.main_root,args.model_dir,args.name)}')
	store.mkdir(parents=True, exist_ok=True)
	reID_bl_csv = f'{store}\\baseline.csv'
	reID_fu_csv = f'{store}\\followup.csv'
	args_txt	= f'{store}\\args.txt'

	cudnn.benchmark = True

#	# import dataset - baseline
	bl_dataset = CheXpertCSV_Dataset(args.image_root, os.path.join(args.main_root,args.dataset_csv), 'baseline')
	#save csv
	img_table = pd.DataFrame(
					bl_dataset.tags,
					columns=['number',	'baseline',	'baseline_patient',	'baseline_study',	'baseline_view', 'match'],
					).to_csv(reID_bl_csv)
	baseline_dataset = Subset(bl_dataset, bl_dataset.numbers, Transform(pixelsize=args.pixelsize))
	baseline_loader = DataLoader(
			baseline_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
	#number of baseline patients
	args.baseline_patients = len(baseline_dataset.indices)
	print(f'baseline_patients: {args.baseline_patients}')

#	# import dataset - followup
	fu_dataset = CheXpertCSV_Dataset(args.image_root, os.path.join(args.main_root,args.dataset_csv), 'followup')
	#save csv
	img_table = pd.DataFrame(
					fu_dataset.tags,
					columns=['number',	'followup',	'followup_patient',	'followup_study',	'followup_view', 'match'],
					).to_csv(reID_fu_csv)
	followup_dataset = Subset(fu_dataset, fu_dataset.numbers, Transform(pixelsize=args.pixelsize))
	followup_loader = DataLoader(
			followup_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
	#患者数
	args.followup_patients =  len(followup_dataset.indices)
	print(f'followup_patients: {args.followup_patients}')

	model = CNN(	model_name=args.arch,  
					pretrained=False,
					px_size=args.pixelsize,
					hide_classes=args.hide_classes,
					n_classes=args.train_patients
					)

	# verify
	test_verification(	
						baseline_loader, 
						followup_loader,
						model, 
						os.path.join(args.main_root,args.model_dir),
						args.model_cpt, 
						f'{store}\\',
						args
						)

