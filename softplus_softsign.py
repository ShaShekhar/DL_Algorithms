import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10)
y1 = (np.exp(x)+1)
y2 = np.log(np.exp(x)+1)
#y3 = x/(np.abs(x)+1)
plt.plot(x,y1,'r',label='without log')
plt.plot(x,y2,'b',label='with log(softplus)')
#plt.plot(x,y3,label='Softsign')
plt.grid()
plt.ylim(-10,10)
plt.legend(loc='lower right')
plt.show()
