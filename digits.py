import struct as st
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    y = np.zeros((len(x),len(x.transpose())))
    for i in range(len(x)):
        for j in range(len(x.transpose())):
            if x[i][j] > 0:
                y[i][j] = 1
    return y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    S = np.exp(x)
    T = np.sum(S, axis = 1, keepdims = True)
    y = S/T
    return y

def d_softmax(x):
    S = softmax(x)
    np.diag(S)
    S_vector = S.reshape(S.shape[0],1)
    S_matrix = np.tile(S_vector,S.shape[0])
    return (np.diag(S) - (S_matrix * np.transpose(S_matrix)))    
    
images_array = pkl.load(open('images_array.pkl','rb'))

labels_array = pkl.load(open('labels_array.pkl','rb'))

N = 60000
n = 10000

#
#Z = images_array[:n]
#Z = (255-Z)/255
#
#X = Z.reshape((n,784))


#X = X - np.mean(X, axis=0)

Y = np.zeros((N,10))

for i in range(N): Y[i][labels_array[i]] = 1


n_in = 784
n_hl1 = 16
n_hl2 = 16
n_out = 10

learning_rate = 0.1

#wh1 = 2*np.random.random((n_in,n_hl1))-1
#bh1 = 2*np.random.random((1,n_hl1))-1
#wh2 = 2*np.random.random((n_hl1,n_hl2))-1
#bh2 = 2*np.random.random((1,n_hl2))-1
#wout = 2*np.random.random((n_hl2, n_out))-1
#bout = 2*np.random.random((1,n_out))-1

wh1 = pkl.load(open('wh1_b.pkl','rb'))
wh2 = pkl.load(open('wh2_b.pkl','rb'))
bh1 = pkl.load(open('bh1_b.pkl','rb'))
bh2 = pkl.load(open('bh2_b.pkl','rb'))
wout = pkl.load(open('wout_b.pkl','rb'))
bout = pkl.load(open('bout_b.pkl','rb'))




for j in range(100):
    for i in range(int(N)):
        #for k in range((i*n):((i+1)*n)):
                
        Z = images_array[i]
        Z = (255-Z)/255
        Z[Z>0] = 1
        X = Z.reshape((1,784))
        
        print(j,i)
        hl1_in = np.dot(X,wh1) + bh1
        hl1_a = sigmoid(hl1_in)

        hl2_in = np.dot(hl1_a,wh2) + bh2
        hl2_a = sigmoid(hl2_in)

        out_in = np.dot(hl2_a,wout) + bout
        out_in = out_in
        out_a = softmax(out_in)
        

        E = Y[i] - out_a

        slope_out = d_sigmoid(out_in)
        slope_hl1 = d_sigmoid(hl1_in)
        slope_hl2 = d_sigmoid(hl2_in)

        #d_out = E * slope_out 
        d_out=np.multiply(E,slope_out)

        Error_at_hl2 = np.dot(d_out, wout.transpose())

        d_hl2 = Error_at_hl2*slope_hl2

        Error_at_hl1 = np.dot(d_hl2, wh2.transpose())

        d_hl1 = Error_at_hl1*slope_hl1

        wout = wout + np.dot(hl2_a.transpose(), d_out)*learning_rate
        wh2 =  wh2 + np.dot(hl1_a.transpose(), d_hl2)*learning_rate
        wh1 = wh1 + np.dot(X.transpose(), d_hl1)*learning_rate

        bout = bout + sum(d_out) * learning_rate
        bh2 = bh2 + sum(d_hl2) * learning_rate
        bh1 = bh1 + sum(d_hl1) * learning_rate

pkl.dump(wh1,open('wh1_b.pkl','wb'))
pkl.dump(wh2,open('wh2_b.pkl','wb'))
pkl.dump(wout,open('wout_b.pkl','wb'))
pkl.dump(bh2,open('bh2_b.pkl','wb'))
pkl.dump(bh1,open('bh1_b.pkl','wb'))
pkl.dump(bout,open('bout_b.pkl','wb'))



        
#hl1 = np.dot(X,wh1) + bh1
#hl1 = 1/(1+np.exp(-hl1))
#hl2 = np.dot(hl1,wh2) + bh2
#hl2 = 1/(1+np.exp(-hl2))
#
#out = np.dot(hl2,wout) + bout
#S = np.exp(out)
#T = np.sum(S, axis = 1, keepdims = True)
#out = S/T
##print(out)
#
#
#max_index = np.argmax(out, axis=1)
##print(max_index)
#
#success = 0
#loss = 0
#
#for i in range(50000,60000):
#    if (labels_array[i] == max_index[i-50000]):
#        success = success+1
#    else:
#        loss = loss+1
#
#efficiency = (success/n)*100
#print(efficiency)