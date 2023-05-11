
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

import shutil
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import timeit

import random

import matplotlib  
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# parserなどで指定
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)


class PadChestCSV_Dataset(Dataset):

	def __init__(self, dir, tsv_file, transform=None):
		self.dir = dir
		self.tags = pd.read_table(tsv_file) #     number,path,position,folder,label,study,view,Sex,Age,No Finding
		self.tags.number   = LabelEncoder().fit_transform(self.tags.number)
		
		self.numbers = self.tags.number
		self.targets = self.tags.label
		self.classes = list(OrderedDict.fromkeys(self.targets))
		self.views   = self.tags.position
		self.ImageDir= self.tags.ImageDir
		self.paths	 = self.tags.ImageID
		self.interval_groups= self.tags.interval_groups
		self.age_groups		= self.tags.age_groups
		self.num_followup	= self.tags.followup
		self.transform		= transform

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		image_id = os.path.join(self.dir, str(self.ImageDir[idx]), self.paths[idx])
		width  = round(self.tags.Columns_DICOM[idx]  * self.tags.SpatialResolution_DICOM[idx])
		height = round(self.tags.Rows_DICOM[idx] * self.tags.SpatialResolution_DICOM[idx])
		image = Image.open(image_id)\
			.convert('L')\
			.resize((width, height), Image.BICUBIC) # 1.0mm per pixel grayscale image

		if self.transform is not None:
			image = self.transform(image)

		label  = self.targets[idx]
		view = self.views[idx]
		interval = self.interval_groups[idx]
		age = self.age_groups[idx]

		return image, label, view, interval, age, image_id

class Transform(object):
	def __init__(
			self,
			pixelsize=[512,512],
		):
		transform = []
		transform.append(transforms.Grayscale(num_output_channels=1))
		transform.append(transforms.CenterCrop(pixelsize))
		transform.append(transforms.ToTensor())
		self.transform = transforms.Compose(transform)
				
	def __call__(self, img):
		return self.transform(img)

class Subset_interval_age(Dataset):
	def __init__(self, dataset, indices, transform=None):
		self.dataset = dataset
		self.age	 = dataset.age_groups
		self.interval   = dataset.interval_groups
		self.views   = dataset.views
		self.paths   = dataset.paths
		self.indices = indices
		self.transform = transform
		subset_labels = lambda items, indices: [item for idx,item in enumerate(items) if idx in indices]
		labels = subset_labels(dataset.targets, indices)
		self.classes = list(OrderedDict.fromkeys(labels))

	def __getitem__(self, idx):
		img, label, view, interval, age, img_id = self.dataset[self.indices[idx]]
		if self.transform:
			img = self.transform(img)
		return img, label, view, interval, age, img_id

	def __len__(self):
		return len(self.indices)

class Batch_size(object):
	def __init__(self, max_batchsize):
		self.max_batchsize = max_batchsize

	def ge_common_divisor(self, arr1, arr2):#greatest_common divisor
		for n in sorted(arr1, reverse=True):
			if n in arr2:
				return n
		return 1

	def valid_divisors(self, max, arr):
		divisors = []
		for n in sorted(arr, reverse=False):
			if max < n:
				return divisors
			else:
				divisors.append(n)
		return divisors

	def make_divisors(self, n):
		lower_divisors , upper_divisors = [], []
		i = 1
		while i*i <= n:
			if n % i == 0:
				lower_divisors.append(i)
				if i != n // i:
					upper_divisors.append(n//i)
			i += 1
		return lower_divisors + upper_divisors[::-1]

	def __call__(self, num_baseline, num_followup):
		bl = self.valid_divisors(self.max_batchsize, self.make_divisors(num_baseline))
		fl = self.valid_divisors(self.max_batchsize, self.make_divisors(num_followup))
		return self.ge_common_divisor(bl, fl)

def eer(fpr,tpr,th):
	""" Returns equal error rate (EER) and the corresponding threshold. """
	fnr = 1-tpr
	abs_diffs = np.abs(fpr - fnr)
	min_idx = np.argmin(abs_diffs)
	eer = np.mean((fpr[min_idx], fnr[min_idx]))
	return eer, th[min_idx]

def test_performance(baseline_loader, followup_loader, model, model_path, model_cpt, store_dir, args):#
	num_features = model.num_features
	score_matrix, label_matrix = [], []
	rank_matrix = []
	target1_matrix, target2_matrix = [], []
	view1_matrix, view2_matrix = [], []
	interval2_matrix, age2_matrix = [], []
	id1_matrix, id2_matrix = [], []
	with torch.no_grad():

		current_model = torch.load(os.path.join(model_path, model_cpt))

		model.load_state_dict(current_model['model'])
		current_epoch = current_model['epoch']

		feature1_vector_list = list()
		feature2_vector_list = list()

		feature_extractor = model.featurier.to(args.device, non_blocking=True)
		store_dir = os.path.join(store_dir, str(model_cpt))
		Path(store_dir).mkdir(exist_ok=True)

		feature_extractor.eval()	# switch to evaluate mode

		for (input2, target2, view2, interval2, age2, id2) in tqdm(followup_loader, total=len(followup_loader), desc=str(current_epoch)+"-"+model_cpt):
			feature2_vector = feature_extractor(input2.to(args.device))
			feature2_vector_list.append(feature2_vector)
			target2_matrix.extend([ i2 for i2 in target2.tolist() ])
			view2_matrix.extend([ i2 for i2 in view2 ])
			interval2_matrix.extend([ i2 for i2 in interval2 ])
			age2_matrix.extend([ i2 for i2 in age2 ])
			id2_matrix.extend([ i2 for i2 in id2 ])
		feature2_vector_list = torch.stack(feature2_vector_list, dim=0).view(-1, num_features)

		for (input1, target1, view1, _, _, id1) in tqdm(baseline_loader, total=len(baseline_loader), desc=str(current_epoch)+"-"+model_cpt):
			feature1_vector = feature_extractor(input1.to(args.device))
			feature1_vector_list.append(feature1_vector)
			target1_matrix.extend([ i1 for i1 in target1.tolist() ])
			view1_matrix.extend([ i1 for i1 in view1 ])
			id1_matrix.extend([ i1 for i1 in id1 ])
		feature1_vector_list = torch.stack(feature1_vector_list, dim=0).view(-1, num_features)

		for j, f2v in tqdm(enumerate(feature2_vector_list), total=len(target2_matrix), desc=str(current_epoch)+"-"+model_cpt):
			f2v_repeat = f2v.repeat((len(target1_matrix), 1))
			score_matrix.insert(	j,
									F.cosine_similarity(
									f2v_repeat, 
									feature1_vector_list,
									dim=1,
									eps=1e-5
									).cpu().detach().numpy().copy()
									)

	genuine  = np.empty(len(target2_matrix))
	impostor = np.empty(len(target1_matrix)*(len(target2_matrix)-1))
	genuine_pair0  = ['']*len(target2_matrix)
	genuine_pair1  = ['']*len(target2_matrix)
	impostor_pair0 = ['']*(len(target1_matrix)*(len(target2_matrix)-1))
	impostor_pair1 = ['']*(len(target1_matrix)*(len(target2_matrix)-1))

	label_matrix = np.zeros((len(target2_matrix),len(target1_matrix)), dtype=np.int16)
	rank_appa	 = np.zeros((5, len(target2_matrix)), dtype=np.int16) #2:AP-AP 1:AP-PA or PA-AP 0:PA-PA
	cmc_appa	 = np.empty((5, len(target1_matrix))) #2:AP-AP 1:AP-PA or PA-AP 0:PA-PA
	appa_str	 = ['']*len(target2_matrix)
	rank_interval	= np.zeros((3, len(target2_matrix)))# < 1yr , 1-5yrs, > 5yrs
	cmc_interval	= np.empty((3, len(target1_matrix))) #2:AP-AP 1:AP-PA or PA-AP 0:PA-PA
	rank_age = np.zeros((5, len(target2_matrix))) # 0, 1-4, 5-10, 11-20, 20-
	cmc_age  = np.empty((5, len(target1_matrix))) #2:AP-AP 1:AP-PA or PA-AP 0:PA-PA

	appa_list =     [str(current_epoch)+"-"+model_cpt, 'PA-PA','PA-AP','AP-AP','AP-PA']#2:AP-AP 1:AP-PA or PA-AP 0:PA-PA
	interval_list = ['lt_1year', '1-5year','gt_5year']
	age_list = 		['neonate-infant(0)', 'young-child(1-4)', 'older-child(5-10)', 'adolescent(11-20)', 'adult(20-)']

	n = 0
	for j, t2 in tqdm(enumerate(target2_matrix), total=len(target2_matrix), desc=str(current_epoch)+"-"+model_cpt):
		for i, t1 in enumerate(target1_matrix):
			if t1 == t2:

				label_matrix[j, i] = 1

				rank_index = np.argsort(-score_matrix[j]).tolist()
				rank_appa[0, j] = rank_index.index(i) + 1				
				rank_interval	[interval2_matrix[j],j] = rank_appa[0, j]
				rank_age		[age2_matrix[j]		,j] = rank_appa[0, j]

				appa_str[j] = view1_matrix[i]+"-"+view2_matrix[j]
				appa_idx = appa_list.index(appa_str[j])
				rank_appa[appa_idx, j] = rank_appa[0,j]

				genuine[j] = score_matrix[j][i]
				genuine_pair0[j],genuine_pair1[j] = id2_matrix[j], id1_matrix[i]
			else:
				impostor[n] = score_matrix[j][i]
				impostor_pair0[n],impostor_pair1[n] = id2_matrix[j], id1_matrix[i]
				n +=1
	impostor_sample_index = random.sample(range(len(impostor.tolist())), len(genuine.tolist()))
	impostor_pair0_sample =  [impostor_pair0[n] for n in impostor_sample_index]
	impostor_pair1_sample =  [impostor_pair1[n] for n in impostor_sample_index]
	genuine_impostor_pair0 = [*genuine_pair0, *impostor_pair0_sample]
	genuine_impostor_pair1 = [*genuine_pair1, *impostor_pair1_sample]
	genuine_impostor_score = [*genuine.tolist(), *impostor[impostor_sample_index].tolist()] # list
	genuine_impostor_label = [*[1]*len(genuine.tolist()),*[0]*len(genuine.tolist())] #list

	for j in range(len(target1_matrix)):
		for i, appa in enumerate(rank_appa):
			cmc_appa[i,j] = np.count_nonzero(appa==(j+1))
			cmc_appa[i,j] += cmc_appa[i,j-1] if j > 0 else 0
		for i, interval in enumerate(rank_interval):
			cmc_interval[i,j] = np.count_nonzero(interval==(j+1))
			cmc_interval[i,j] += cmc_interval[i,j-1] if j > 0 else 0
		for i, age in enumerate(rank_age):
			cmc_age[i,j] = np.count_nonzero(age==(j+1))
			cmc_age[i,j] += cmc_age[i,j-1] if j > 0 else 0

	#rank.csv
	rank_csv = f'{store_dir}\\rank.tsv'
	with open(rank_csv, 'w') as f_handle:
		np.savetxt(f_handle, [np.hstack(('series'	, np.array(target2_matrix)))], delimiter='\t', fmt="%s")
		np.savetxt(f_handle, [np.hstack(('score'	, np.array(genuine)))], delimiter='\t', fmt="%s")
		np.savetxt(f_handle, [np.hstack(('position'	, np.array(appa_str)))], delimiter='\t', fmt="%s")
		np.savetxt(f_handle, [np.hstack(('interval'	, np.array(interval2_matrix)))], delimiter='\t', fmt="%s")
		np.savetxt(f_handle, [np.hstack(('age'		, np.array(age2_matrix)))], delimiter='\t', fmt="%s")

		for i, appa in enumerate(rank_appa):
			np.savetxt(f_handle, [np.hstack((appa_list[i], appa))], delimiter='\t', fmt="%s")
		for i, interval in enumerate(rank_interval):
			np.savetxt(f_handle, [np.hstack((interval_list[i], interval))], delimiter='\t', fmt="%s")
		for i, age in enumerate(rank_age):
			np.savetxt(f_handle, [np.hstack((age_list[i], age))], delimiter='\t', fmt="%s")

		np.savetxt(f_handle, [''], delimiter='\t', fmt="%s")#blank row
		np.savetxt(f_handle, [np.hstack(('rank', [1,2,5,10,'Total']))], delimiter='\t', fmt="%s")

		for r in range(6,18):
			np.savetxt(f_handle, [np.hstack(('=A%d' %(r), [
													'=COUNTIFS(%d:%d,"<="&B$20,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,r,r,r,r),
													'=COUNTIFS(%d:%d,"<="&C$20,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,r,r,r,r),
													'=COUNTIFS(%d:%d,"<="&D$20,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,r,r,r,r),
													'=COUNTIFS(%d:%d,"<="&E$20,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,r,r,r,r),
													'=COUNTIF(%d:%d,">0.1")' %(r,r),
													]))], delimiter='\t', fmt="%s")

	#cmc.csv
	cmc_csv = f'{store_dir}\\cmc.tsv'
	with open(cmc_csv, 'w') as f_handle:
		np.savetxt(f_handle, [np.hstack(('series', range(1, args.baseline_patients[0]+1)))], delimiter='\t', fmt="%s")
		for i, appa in enumerate(cmc_appa):
			np.savetxt(f_handle, [np.hstack((appa_list[i], appa))], delimiter='\t', fmt="%s")
		for i, interval in enumerate(cmc_interval):
			np.savetxt(f_handle, [np.hstack((interval_list[i], interval))], delimiter='\t', fmt="%s")
		for i, age in enumerate(cmc_age):
			np.savetxt(f_handle, [np.hstack((age_list[i], age))], delimiter='\t', fmt="%s")

	#pROC.tsv
	proc_tsv		= '%s\\pROC.tsv' %(store_dir)
	pd.DataFrame({
					'score': genuine_impostor_score, 
					'genuine': genuine_impostor_label,
					'pair0': genuine_impostor_pair0,
					'pair1': genuine_impostor_pair1,
				}).to_csv(proc_tsv, sep='\t', index=False)

	#AUC ROC
	roc_curve_png = f'{store_dir}\\roc_curve.png'
	fpr, tpr, th = roc_curve(genuine_impostor_label, genuine_impostor_score)
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
	ax.hist(genuine,  bins=100, range=(0, 1), density=True, alpha=0.7)
	ax.hist(impostor[impostor_sample_index], bins=100, range=(0, 1), density=True, alpha=0.7)
	ax.set_xlabel('Cosine Similarity')
	ax.set_ylabel('Relative Frequency')
	plt.title('Genuine-Impostor histogram EER:%.4f %.4f' %(eer_value, eer_pos))
	#plt.show()
	plt.savefig('%s' %(histogram_png))
	plt.clf()
	plt.close()

	return_value = [current_epoch]
	return_value.append(auc_value)
	return_value.append(eer_value)

	#show acc1 
	result_txt = f'{store_dir}\\result.csv'
	with open(result_txt, 'w') as f:
		print('%s,%s' %('model_cpt',        model_cpt), file=f)
		print('%s,%s' %('current epoch',    current_epoch), file=f)
		print('%s,%f' %('AUC',              auc_value), file=f)
		print('%s,%f' %('EER',              eer_value), file=f)
		print('%s,%f' %('EER point',        eer_pos), file=f)
		print('%s,%f' %('Genuine(mean)',    np.average(genuine)), file=f)
		print('%s,%f' %('Impostor(mean)',   np.average(impostor[impostor_sample_index])), file=f)

		print('%s %s' %(model_cpt.split('.')[0], current_epoch))
		print('AUC %.4f' %(auc_value))
		print('EER %.4f' %(eer_value))
		for i, appa in enumerate(cmc_appa):
			return_value.append(appa[0]/appa[-1])
			print('%s: R1 %.4f R2 %.4f' %(appa_list[i], appa[0]/appa[-1], appa[1]/appa[-1]))
			print('%s,%f,%f' %(appa_list[i], appa[0]/appa[-1], appa[1]/appa[-1]), file=f)
		return_value.append(cmc_appa[0][1]/cmc_appa[0][-1])
		for i, interval in enumerate(cmc_interval):
			return_value.append(interval[0]/interval[-1])
			print('%s: R1 %.4f' %(interval_list[i], interval[0]/interval[-1]))
			print('%s,%f' %(interval_list[i], interval[0]/interval[-1]), file=f)
		for i, age in enumerate(cmc_age):
			return_value.append(age[0]/age[-1])
			print('%s: R1 %.4f' %(age_list[i], age[0]/age[-1]))
			print('%s,%f' %(age_list[i], age[0]/age[-1]), file=f)

	return return_value

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--name', default='PadChest',	help='model name: save folder')
	parser.add_argument('--main_root', default='./')
	parser.add_argument('--model_dir', default='models/dropout-5e-01_ep-300_px-320')
	parser.add_argument('--model_cpt', default='bestmodel.cpt')
	parser.add_argument('--image_root', default=r'.\images\PadChest')
	parser.add_argument('--dataset_csv', default=r'.\images\PadChest.txt')
	parser.add_argument('--pixelsize', default=[512,512], type=int) #PadChest
	parser.add_argument('--arch', default='tf_efficientnetv2_s')
	parser.add_argument('--train_patients', default=3245, type=int, help='patient num. on training')
	parser.add_argument('--baseline_patients', default=0, type=int, help='patient num. verification')
	parser.add_argument('--followup_patients', default=0, type=int, help='patient num. verification')
	parser.add_argument('--num_features', default=1280, type=int, help='dimention of embedded features')
	parser.add_argument('--hide_classes', default=1024, type=int)
	parser.add_argument('-b', '--batch_size', default=128,	type=int,	metavar='N', help='mini-batch size') 
	parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
	parser.add_argument('--pin_memory', default=True, type=bool)
	parser.add_argument('--drop_last', default=False,	type=bool)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args()

	warnings.simplefilter('ignore')

	store = Path(f'{os.path.join(args.main_root,args.model_dir,args.name)}')
	store.mkdir(parents=True, exist_ok=True)
	baseline_csv = f'{store}\\baseline.csv'
	followup_csv = f'{store}\\followup.csv'
	args_txt	= f'{store}\\args.txt'
	shutil.copyfile(os.path.join(args.main_root,args.dataset_csv), os.path.join(f'{store}', args.dataset_csv.rsplit('\\', 1)[1]))
	
	#print('Config -----')
	#for arg in vars(args):
	#	print('%s: %s' %(arg, getattr(args, arg)))
	#print('------------')

	with open(args_txt, 'w') as f:
		for arg in vars(args):
			print('%s: %s' %(arg, getattr(args, arg)), file=f)

#	# import dataset
#################################################################################
	all_dataset = PadChestCSV_Dataset(args.image_root, os.path.join(args.main_root,args.dataset_csv), None) 

	bl_subset = all_dataset.numbers[all_dataset.num_followup==0].tolist()
	fu_subset = all_dataset.numbers[all_dataset.num_followup> 0].tolist()

	bl_dataset = Subset_interval_age(all_dataset, bl_subset, Transform(pixelsize=args.pixelsize))
	fu_dataset = Subset_interval_age(all_dataset, fu_subset, Transform(pixelsize=args.pixelsize))

	#number of patients
	args.baseline_patients = len(bl_dataset.classes), len(bl_dataset.indices)
	print(f'baseline_patients: {args.baseline_patients}')
	args.followup_patients = len(fu_dataset.classes), len(fu_dataset.indices)
	print(f'followup_patients: {args.followup_patients}')

	#calc batch_number
	print(str(args.batch_size), end=" -> ")
	MyBatch = Batch_size(args.batch_size)
	args.batch_size = MyBatch(args.baseline_patients[1],args.followup_patients[1])
	print(str(args.batch_size))

	img_table = pd.DataFrame(
					all_dataset.tags,
					columns=all_dataset.tags.columns[1:],
					)
	img_table.loc[fu_subset].to_csv(followup_csv)
	img_table.loc[bl_subset].to_csv(baseline_csv)

# worker_init, g : seed fixing
	baseline_loader = DataLoader(
			bl_dataset, batch_size=args.batch_size,	shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last, worker_init_fn=seed_worker, generator=g)
	followup_loader = DataLoader(
			fu_dataset, batch_size=args.batch_size,	shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last, worker_init_fn=seed_worker, generator=g)

	model = CNN(	model_name=f'{args.arch}',  
					pretrained=False,
					px_size=args.pixelsize,
					n_classes=args.train_patients,
					hide_classes=args.hide_classes, 
					)

	# verification, identification
	checkpoint = [args.model_cpt]
	print(checkpoint)

	shutil.copyfile(f'{store}\\result.csv', f'{store}\\result_temp.csv') if os.path.exists(f'{store}\\result.csv') else ''
	if not os.path.exists(f'{store}\\result.csv'):
		with open(f'{store}\\result.csv', 'w') as f:
			print('epoch,best_epoch,auc,eer,acc1-all,acc1-PAPA,acc1-PAAP,acc1-APAP,acc1-APPA,acc2-all,lt_1year,1-5year,gt_5year,neonate-infant(0),young-child(1-4),older-child(5-10),adolescent(11-20),adult(20-),time', file=f)

	for cpt in checkpoint:
		tic=timeit.default_timer()
		best_epoch, auc_v, eer_v,\
			acc1_all, acc1_papa, acc1_appa, acc1_paap, acc1_apap, acc2_all,\
			acc1_i0, acc1_i1,acc1_i2, acc1_i3, acc1_i4, acc1_a0, acc1_a1, acc1_a2\
			= test_performance(
								baseline_loader, followup_loader, 
								model, os.path.join(args.main_root,args.model_dir),	cpt, 
								f'{store}\\', args)
			
		elapse_time = int(timeit.default_timer() - tic)
		print(elapse_time)
		with open(f'{store}\\result.csv', 'a') as f_handle:
			print('%s,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d' \
				%(cpt.split('.')[0], best_epoch, auc_v, eer_v, acc1_all, acc1_papa, acc1_appa, acc1_paap, acc1_apap, acc2_all, acc1_i0, acc1_i1,acc1_i2, acc1_i3, acc1_i4, acc1_a0, acc1_a1, acc1_a2, elapse_time), file=f_handle)
