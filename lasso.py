# Importing libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from RegGen import *

import seaborn as sns

# Lasso Regression

class LassoRegression() :
	
	def __init__( self, learning_rate, iterations, l1_penality ) :
		
		self.learning_rate = learning_rate
		
		self.iterations = iterations
		
		self.l1_penality = l1_penality

		self.plt_learning_rate = dict()
		self.plt_W = dict()
		
	# Function for model training
			
	def fit( self, X, Y ) :
		
		# no_of_training_examples, no_of_features
		
		self.m, self.n = X.shape
		
		# weight initialization
		
		self.W = np.zeros( self.n )
		
		self.b = 0
		
		self.X = X
		
		self.Y = Y
		
		# gradient descent learning
				
		for i in range( self.iterations ) :
			
			self.update_weights()
			self.plt_W[i]=self.W
			

		return self
	
	# Helper function to update weights in gradient descent
	
	def update_weights( self ) :
			
		Y_pred = self.predict( self.X )
		
		# calculate gradients
		
		dW = np.zeros( self.n )
		
		for j in range( self.n ) :
			
			if self.W[j] > 0 :
				
				dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )
						
						+ self.l1_penality ) / self.m

		
			else :
				
				dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )
						
						- self.l1_penality ) / self.m

	
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m
		
		# update weights
	
		self.W = self.W - self.learning_rate * dW
	
		self.b = self.b - self.learning_rate * db
		
		return self
	
	# Hypothetical function h( x )
	
	def predict( self, X ) :
	
		return X.dot( self.W ) + self.b
	

def main() :
	
	# Importing dataset
	colors = sns.color_palette("husl", 10)

	df = pd.read_csv( ".\\generated\\sample_size_473.csv" )
	# fig, ax = plt.subplots(2)
	# for i,c in enumerate(list(df.columns)):
 #            df.plot(ax=ax, kind='scatter', x=c, y='y',color=colors[i],label=c)
	cols = [ c for c in df.columns if 'x' in c]
	X = df[[*cols]].values
	Y = df['y'].values
	print('X',X.shape,X)
	print('y',Y.shape,Y)
	# Splitting dataset into train and test set

	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1 / 3, random_state = 0 )
	
	print(X_train.shape,len(X_train))
	# Model training
	
	model = LassoRegression( iterations = 1000, learning_rate = 0.01, l1_penality = 500 )

	model.fit( X_train, Y_train )
	
	# Prediction on test set

	Y_pred = model.predict( X_test )
	print('----------------------------')
	print(len(Y_pred))
	print(Y_test)
	print( "Predicted values ", np.round( Y_pred[:3], 2 ) )
	
	print( "Real values	 ", Y_test[:3] )
	
	print( "Trained W	 ", round( model.W[0], 2 ) )
	
	print( "Trained b	 ", round( model.b, 2 ) )
	
	# Visualization on test set
	print(type(X_test),type(Y_test))
	pltdf=pd.DataFrame(X_test,columns=['x_'+str(i) for i in range(X_test.shape[1]) ])
	pltdf['y'] = Y_test
	fig, ax = plt.subplots(2)
	for i,c in enumerate(list(cols)):
            pltdf.plot(ax=ax[0], kind='scatter', x=c, y='y',color=colors[i],label=c)
	# plt.scatter( X_test, Y_test, color = 'blue' )
	pltdf['y_pred']=Y_pred
	# for i,c in enumerate(cols):
	pltdf.plot(ax=ax[0], kind='scatter',x='x_0', y='y_pred',color='orange',label='predicted')	
	# plt.plot( X_test, Y_pred, color = 'orange' )
	plt.title( 'Lasso Regression' )
	
	plt.xlabel( 'X' )

	plt.ylabel( 'Y' )
	diff_lr = abs(min(model.plt_learning_rate.values()) - max(model.plt_learning_rate.values()))
	aug_lr = [diff_lr*10*lr for lr in model.plt_learning_rate.values()]

	# ax[1].plot(model.plt_learning_rate.values())
	lr_pltdf=pd.DataFrame(list(model.plt_learning_rate.items()),columns=['iteration','learning_rate'] )
	lr_pltdf.plot(ax=ax[1], kind='scatter',x='iteration', y='learning_rate',color='red',label='lambdas')
	plt.title( 'Lasso Lambdas' )
	plt.text(10, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
	plt.xlabel( 'itr' )

	plt.ylabel( 'Learning Rate (Lamda)' )
	
	plt.show()


if __name__ == "__main__" :
	
	main()
