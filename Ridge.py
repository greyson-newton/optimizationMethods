# Importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Ridge Regression

class RidgeRegression() :
	
	def __init__( self, learning_rate, iterations, l2_penality ) :
		
		self.learning_rate = learning_rate		
		self.iterations = iterations		
		self.l2_penality = l2_penality
		
	# Function for model training			
	def fit( self, X, Y ) :
		print(X.shape)
		# no_of_training_examples, no_of_features		
		shape = X.shape
		self.m = shape[0]
		if len(shape) == 1:
			self.n=1
		else:
			self.n=shape[1]
		# weight initialization		
		self.W = np.zeros( self.n )
		
		self.b = 0		
		self.X = X		
		self.Y = Y
		
		# gradient descent learning
				
		for i in range( self.iterations ) :			
			self.update_weights()			
		return self
	
	# Helper function to update weights in gradient descent
	
	def update_weights( self ) :		
		Y_pred = self.predict( self.X )
		
		# calculate gradients	
		dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +			
			( 2 * self.l2_penality * self.W ) ) / self.m	
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m
		
		# update weights	
		self.W = self.W - self.learning_rate * dW	
		self.b = self.b - self.learning_rate * db		
		return self
	
	# Hypothetical function h( x )
	def predict( self, X ) :	
		return X.dot( self.W ) + self.b
	
# Driver code

def main() :
	
	# Importing dataset	
	df = pd.read_csv( ".\\generated\\sample_size_473.csv" )
	cols = [ c for c in df.columns if 'x' in c]
	X = df[[*cols]].values
	Y = df['y'].values

	# Splitting dataset into train and test set
	X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
											
										test_size = 1 / 3, random_state = 0 )
	
	# Model training	
	model = RidgeRegression( iterations = 1000,							
							learning_rate = 0.01, l2_penality = 1 )
	model.fit( X_train, Y_train )
	
	# Prediction on test set
	Y_pred = model.predict( X_test )	
	print( "Predicted values ", np.round( Y_pred[:3], 2 ) )	
	print( "Real values	 ", Y_test[:3] )	
	print( "Trained W	 ", round( model.W[0], 2 ) )	
	print( "Trained b	 ", round( model.b, 2 ) )
	
	# Visualization on test set	
	colors = sns.color_palette("husl", 10)
	pltdf=pd.DataFrame(X_test,columns=['x_'+str(i) for i in range(X_test.shape[1]) ])
	pltdf['y'] = Y_test
	fig, ax = plt.subplots()
	for i,c in enumerate(cols):
            pltdf.plot(ax=ax, kind='scatter', x=c, y='y',color=colors[i],label=c)
	# plt.scatter( X_test, Y_test, color = 'blue' )
	pltdf['y_pred']=Y_pred
	# for i,c in enumerate(cols):
	pltdf.plot(ax=ax, kind='scatter',x='x_0', y='y_pred',color='orange',label='predicted')	
	plt.title( 'Ridge Regression' )
	
	plt.xlabel( 'X' )
	
	plt.ylabel( 'Y' )	
	plt.show()
	
if __name__ == "__main__" :
	main()
