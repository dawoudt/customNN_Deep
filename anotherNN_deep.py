import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

def predict(xtest, ytest):
	a1 = reLU(np.dot(xtest, W) + b)
	a2 = reLU(np.dot(a1, W1) + b)
	scores = np.dot(a2, W2) + b2
	predicted_class = np.argmax(scores, axis=1)
	print("\npredicted:\n", predicted_class, "\n\nActual:\n", ytest)
	print ('training accuracy: %.2f' % (np.mean(predicted_class == ytest)))


def generate_paramaters(inp, out):
	# when using a large data set, be sure to reduce variance in weights with np.sqrt(2.0/inp)
		W = (0.1 * (np.random.randn(inp,out) / np.sqrt(2.0/inp) )) 
		b = np.ones((1,out))
		mom_w = np.zeros((inp, out))
		mom_b = np.zeros((1, out))
		return W, b, mom_w, mom_b


def reLU(preactivation):
	return np.maximum(0, preactivation)


# df = pd.read_csv('./iris.data', names=['SW', 'SL', 'PW', 'PL',  'Label' ])

# df = df.reindex(np.random.permutation(df.index))

# XX = df.ix[:,:4]
# yy = df.ix[:,4:5]

# X = XX.as_matrix()
# X = X/np.amax(X, axis=0)

# yy = yy['Label'].astype('category')
# y = yy.cat.codes.as_matrix()

# X_train = X[:100]
# y_train = y[:100]

# X_test = X[100:]
# y_test = y[100:]


df = pd.read_csv('./breast-cancer-wisconsin.data', names = ['Sample code number', 'clump thickness', 'uniformity of cell size', \
														'uniformity of cell shape', 'marginal adhesion', 'single epithelial cell size',\
														'bare nuclei','bland chromatin','normal nucleoli','mitoses','class'] )

df = df.reindex(np.random.permutation(df.index))

data_set = df.ix[:,:10]
class_set = df.ix[:,10:11]

data_set.drop(['Sample code number'], axis = 1, inplace = True)

data_set['bare nuclei'] = data_set['bare nuclei'].astype('category')
data_set['bare nuclei'] = data_set['bare nuclei'].cat.codes

X = data_set.as_matrix()
X = X/np.amax(X, axis=0) 

yy = class_set['class'].astype('category')
y = yy.cat.codes.as_matrix()


X_train = X[:600]
y_train = y[:600]


X_test = X[600:]
y_test = y[600:]





# loss_list = []

# N = 10 # number of points per class
D = X_train.shape[1] # dimensionality
K = np.unique(y).shape[0]

# np.unique(y).shape[0]# number of classes

# initialize parameters randomly
h = 100 # size of hidden layer


W, b, mom_w, mom_b = generate_paramaters(D, h)
W1, b1, mom_w1, mom_b1 = generate_paramaters(h, h)
W2, b2, mom_w2, mom_b2 = generate_paramaters(h, K)


# some hyperparameters
step_size = 1e-2
momentum = 0.95
dropout_percentage = 0.2
dropout_flag = False
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X_train.shape[0]




for i in range(10000):


	
	# evaluate class scores, [N x K]
	z1 = np.dot(X_train, W) + b
	a1 = reLU(z1) # note, ReLU activation
	if dropout_flag:
		a1*= np.random.binomial([np.ones((len(X_train),h))], 1-dropout_percentage )[0] * (1.0/(1-dropout_percentage))

	z2 = np.dot(a1, W1) + b1
	a2 = reLU(z2)
	if dropout_flag:
		a2*= np.random.binomial([np.ones((len(X_train),h))], 1-dropout_percentage )[0] * (1.0/(1-dropout_percentage))

	scores = np.dot(a2, W2)

	# compute the class probabilities
	exp_scores = np.exp(scores - np.max(scores))
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]



	# compute the loss: average cross-entropy loss and regularization
	correct_logprobs = log_loss(y_train, (probs[range(num_examples)]))
	# correct_logprobs = -np.log(probs[range(num_examples),y_train])
	data_loss = np.sum(correct_logprobs)/num_examples
	reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) 
	loss = data_loss + reg_loss

	
	if i % 1000 == 0:
		print("\niteration %d: loss %f" % (i, loss))
		predict(X_test, y_test)

	# compute the gradient on scores
	dscores = probs
	dscores[range(num_examples),y_train] -= 1
	dscores /= num_examples


	
	# backpropate the gradient to the parameters
	# first backprop into parameters W2 and b2
	dW2 = np.dot(a2.T, dscores)
	db2 = np.sum(dscores, axis=0, keepdims=True)
	# next backprop into hidden layer
	d_a2 = np.dot(dscores, W2.T)
	# backprop the ReLU non-linearity
	d_a2[a2 <= 0] = 0

	dW1 = np.dot(a1.T, d_a2)
	db1 = np.sum(d_a2, axis=0, keepdims=True)

	d_a1 = np.dot(d_a2, W1.T)
	d_a1[a1 <= 0] = 0

	# finally into W,b
	dW = np.dot(X_train.T, d_a1)
	db = np.sum(d_a1, axis=0, keepdims=True)
	
	# add regularization gradient contribution
	dW2 += reg * W2
	dW1 += reg * W1
	dW += reg * W


	# update deltas with momentum factor
	new_dW2 = dW2 + momentum * mom_w2
	new_db2 = db2 + momentum * mom_b2

	new_dW1 = dW1 + momentum * mom_w1
	new_db1 = db1 + momentum * mom_b1

	new_dW = dW + momentum * mom_w
	new_db = db + momentum * mom_b
	
	# update weights matrix
	W += -step_size * new_dW
	b += -step_size * new_db

	W1 += -step_size * new_dW1
	b1 += -step_size * new_db1

	W2 += -step_size * new_dW2
	b2 += -step_size * new_db2

	# update without momentum
	# W += -step_size * dW
	# b += -step_size * db

	# W2 += -step_size * dW2
	# b2 += -step_size * db2

	mom_w2 = new_dW2
	mom_b2 = new_db2

	mom_w1 = new_dW1
	mom_b1 = new_db1

	mom_w = new_dW
	mom_b = new_db






