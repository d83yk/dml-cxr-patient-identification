from torch import nn
import torch.nn.functional as F
import timm

#forked from https://github.com/4uiiurz1/pytorch-adacos
# ./pytorch-adacos-master/metrics.py -> ./metrics.py
from metrics import AdaCos

class Network(nn.Module):
	def __init__(self, model_name, pretrained):
		super(Network, self).__init__()
		self.in_channels=1
		self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=self.in_channels, num_classes=0)
		self.num_features = self.backbone.num_features

	def forward(self, x):
		x = self.backbone(x)
		x = x.view(x.size(0), -1)
		return x

class Featurier(nn.Module):
	def __init__(self, model_name, pretrained, px_size):
		super(Featurier, self).__init__()
		model = Network(model_name=model_name, pretrained=pretrained)
		self.backbone = model.backbone
		self.px_size = px_size
		self.num_features = model.num_features
		self.in_channels = model.in_channels

	def forward(self, x):
		x = x.expand(x.data.shape[0], self.in_channels, self.px_size[0], self.px_size[1])
		x = self.backbone(x)
		x = x.view(x.size(0), -1)
		return x

class Classifier(nn.Module):
	def __init__(self, num_features, n_classes, hide_classes, dropout):
		super(Classifier, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(num_features, hide_classes),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hide_classes, num_features)
			)
		self.metric = AdaCos(num_features=num_features, num_classes=n_classes)

	def forward(self, x, t):
		x = self.fc(x)
		x = self.metric(x, t)
		return x

class CNN(nn.Module):
	def __init__(self, model_name='tf_efficientnetv2_s', pretrained=False, px_size=[320,320], n_classes=3245, hide_classes=1024, drop_out=0.5):
		super(CNN, self).__init__()
		self.featurier  = Featurier(model_name=model_name, 
			                        pretrained=pretrained, 
				                    px_size=px_size)
		self.classifier = Classifier(num_features=self.featurier.num_features, 
			                        n_classes=n_classes, 
				                    hide_classes=hide_classes, 
				                    dropout=drop_out)
		self.num_features = self.featurier.num_features

	def forward(self, x, t):
		x = self.featurier(x)
		x = self.classifier(x, t)
		return x
