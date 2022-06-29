import numpy as np

class NueralNetwork:
	def __init__(self,layers,a=0.1):
		self.W=[]
		'''layers is the list of integers which represents structure'''
		self.layers=layers
		self.a=a
		for i in np.arange(0,len(layers)-2):
			w=np.random.rand(layers[i]+1,layers[i+1]+1)
			self.W.append(w/np.sqrt(layers[i]))
		w=np.randon.rand(layers[-2]+1,layers[-1])
		self.W.append(w/np.sqrt(layers[-2]))
	
	def __repr__(self):
		return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
	
	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))
	
	def sigmoid_derivative(self,x):
		return x*(1-x)
	
	def fit(self,X,y,epochs=1000,dis):
		X=np.c_[X,np.ones((X.shape[0]))]
		for epoch in np.arange(0,epochs):
			for (x,target) in zip(X,y):
				self.fit_partial(x,target)
			
			if epoch==0 or (epoch+1)%dis==0:
				loss=self.calculate_loss(X,y)
				print("[INFO] epoch={},loss={:.7f}".format(epoch+1,loss))
	def fit_partial(self,X,y):
		A=[np.atleast_2d(x)]
		for layer in np.arange(0,len(self.W)):
			feed=A[layer].dot(self.W[layer])
			out=self.sigmoid(feed)
			A.append(out)

		error=A[-1]-y

		D=[error*self.sigmoid_derivative(A[-1])]
		for layer in np.arange(len(A)-2,0,-1):
			delta=D[-1].dot(self.W[layer].T)
			delta=delta*self.sigmoid_derivative(A[layer])
			D.append(delta)

		D=D[::-1]
		for layer in np.arange(0,len(self.W)):
			self.W[layer]+=-self.a*A[layer].T.dot(D[layer])

	def predict(self,X,bias=True):
		p=np.atleast_2d(X)
		if bias:
			p=np.c_[p,np.ones((p.shape[0]))
		for layer in np.arange(0,len(self.W)):
			p=self.sigmoid(np.dot(p,self.W[layer]))
	
	return p
	
	def calculate_loss(self,X,target):
		targets=np.atleast_2d(targets)
		pred=self.predict(X,bias=True)
		loss=0.5*np.sum((pred-targets)**2)
		return loss

