#importing numpy python library
import numpy as np
#defining a sigmoid function you can use inbuilt np. Sign 
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x) #this is a derivative of signum function. This need to calculate gradient decent
    return 1/(1+np.exp(-x))# this is a signum function
#input.. It can be any truth table 
X = np.array([ 
	            [0,0,1],
	            [0,1,0],
	            [0,1,1],
	            [1,0,0],
	            [1,0,1],
	            [1,1,0],
	            [1,1,1]
	               ])
#adding bias so that for 0 input our programme doesn't make wrong prediction
X=np.c_[np.ones(len(X)), X] 
#y is the output of the truth table
y = np.array([1,1,0,1,0,0,0]).T
#seed is done to create the same random number so that it is easy to analyze our result
np.random.seed(1)
#creating random weights
syn0 = 2*np.random.random((len(X[0])))-1 
# this for loop is for training our model here we use 10000 cycle in each cycle it will predict a output and then see error and update weights to predict correct next time
for iter in xrange(10000):
    l0 = X 
    l1 = nonlin(np.dot(l0,syn0)) #predicted output
    l1_error = y - l1 #calculating error
    l1_delta = l1_error * nonlin(l1,True) #calculating gradient for gradient decent 
    syn0 += np.dot(l0.T,l1_delta) #updating weights
print "Output After Training:  "
print l1 #predicting output for training set as input
