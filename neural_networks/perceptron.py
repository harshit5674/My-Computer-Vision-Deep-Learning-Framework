import numpy as np

class Perceptron:
	def __init__(self,n,a=0.1):
		self.W=np.random.rand(n+1)/np.sqrt(n)
		self.a=a
	
	def step(self,x):
		return 1 if x>0 else 0
	
	def fit(self,X,y,epochs=10):
		X=np.c_[X,np.ones((X.shape[0]))]
		for epoch in np.arange(0,epochs):
			for(x,target) in zip(X,y):
				p=self.step(np.dot(x,self.W))
				if p!=target:
					error=p-target
					self.W+=-self.a*error*x
	
	def predict(self,X,addBias=True):
		X=np.atleast_2d(X)
		if addBias:
			X=np.c_[X,np.ones((X.shape[0]))]
		return self.step(np.dot(X,self.W))



