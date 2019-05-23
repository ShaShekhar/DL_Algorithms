import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10,0.1)
y_tanh = np.tanh(x) #(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
y_sigmoid = 1.0/(1.0 + np.exp(-x))
plt.figure(1)
# Tanh gragh
#plt.subplot(1,2,1)
#plt.title("tanh Graph")
#plt.plot(x,y_tanh)
# Inside a subplot both tanh and sigmoid graph
#plt.subplot(1,2,2)
#plt.title("Sigmoid and tanh Graph")
plt.title("Sigmoid Graph")
plt.plot(x,y_tanh)
plt.plot(x,y_sigmoid)
plt.xlabel("Value of x")
plt.ylabel("Value of y")
plt.ylim(-1.1,1.1)
plt.grid()
#plt.title("Sine Graph")
#plt.legend(["tanh","sigmoid"])
plt.show()
