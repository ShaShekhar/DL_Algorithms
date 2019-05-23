import numpy as np
h1 = np.array([1,2,3,4,5,6,7,8,9,10],dtype="float32") #[ 1  2  3  4  5  6  7  8  9 10]
print(h1)
#drop1 = np.random.rand(*h1.shape) < 0.5 #[ True  True False  True  True False  True False False  True]
drop1 = (np.random.rand(*h1.shape) < 0.5)/0.5 #To Boost the performance at test time increase the activation
#[ 0.  2.  2.  2.  2.  2.  0.  0.  0.  2.]
print(drop1)
h1 *= drop1
print(h1) #[ 1  2  0  4  5  0  7  0  0 10] & for boosted activation [  0.   4.   6.   8.  10.  12.   0.   0.   0.  20.]
