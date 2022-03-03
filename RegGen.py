import numpy as np
import pandas as pd
class RegGen():
	def __init__(self,n_samples=20, n_features=4, n_informative=2, n_targets=1, 
	                        bias=0.0, effective_rank=None,tail_strength=0.5, 
	                        noise=0.0, shuffle=True, coef=True, random_state=True):
		self.n_samples=n_samples
		self.n_features=n_features
		self.n_informative=n_informative
		self.n_targets=n_targets

		self.bias=bias
		self.effective_rank=effective_rank
		self.tail_strength=tail_strength

		self.noise=noise
		self.shuffle=shuffle
		self.coef=coef
		self.random_state=random_state
		self.attributes = {"n_samples":self.n_samples,
		"n_features":self.n_features, 
		"n_informative":self.n_informative, 
		"n_targets":self.n_targets,
		"bias":self.bias,
		"effective_rank":self.effective_rank,
		"tail_strength":self.tail_strength,
		"noise":self.noise,
		"shuffle":self.shuffle,
		"coef":self.coef,
		"random_state":self.random_state,
		}
		# self.attributes = {v: k for k, v in self.attributes.items()}
		self.distributions = dict()
	def set_attr (self,attr_dict):

		for key,item in attr_dict.items():
			# print(self.attributes[key])
			self.attributes[key]=item
			# print(self.attributes[key])
	def generate (self,name=''):
		from sklearn.datasets import make_regression
		print('GENERSTE',self.attributes.values())
		data1 = make_regression(*self.attributes.values())
		# print(data1)
		
		df1 = pd.DataFrame(data1[0],columns=['x_'+str(i) for i in range(data1[0].shape[1]) ])
		df1['y'] = data1[1]
		# self.distributions[name]=df1
		return df1
	def ret_xy(self,name):
		df=self.distributions.get(name)
		return [column for column in df.columns if 'x' in column],
		[column for column in df.columns if 'y' in column]