import numpy as np
import matplotlib.pyplot as plt
# Assume some unit gaussian 10-D input data
D = np.random.randn(1000,500)
hidden_layer_sizes = [500]*10 #[500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
nonlinearities = ["tanh"]*len(hidden_layer_sizes)
act = {"relu":lambda x:np.maximum(0,x), "tanh":lambda x:np.tanh(x)}
Hs = {}
for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i-1] #input at this layer
    print("X:",X.shape)
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.randn(fan_in,fan_out)*0.01#/np.sqrt(fan_in) # layer initialization
    print("W:",W.shape)
    H = np.dot(X,W) # matrix multiply
    print(H.shape)
    H = act[nonlinearities[i]](H) # nonlinearilty
    Hs[i] = H # cache result on this layer
# loook at distribution at each layer
print("input layer had mean {} and std {}".format(np.mean(D),np.std(D)))
layer_means = [np.mean(H) for i,H in Hs.items()]
layer_stds = [np.std(H) for i,H in Hs.items()]
for i,H in Hs.items():
    print("hidden layer {} had mean {} and std {}".format(i+1,layer_means[i],layer_stds[i]))
# plot the mean and standard deviation
plt.figure()
plt.subplot(121)
plt.plot(Hs.keys(),layer_means,"ob-")
plt.title("layer mean")
plt.subplot(122)
plt.plot(Hs.keys(),layer_stds,"or-")
plt.title("layer std")
#plt.show()

#plot the distribution
plt.figure()
for i,H in Hs.items():
    plt.subplot(1,len(Hs),i+1)
    plt.hist(H.ravel(),30,range(-1,1))
plt.show()
