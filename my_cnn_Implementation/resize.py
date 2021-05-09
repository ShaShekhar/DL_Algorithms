import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from scipy.misc import imread,imresize,imsave
import matplotlib.pyplot as plt

sess = tf.Session()
#img = Image.open('GoldFish.jpg')
#img = imread('GoldFish.jpg')
#print(img.shape)
#img = img.thumbnail((250,250))
#img.save('resize_gf.jpg')
img = cv2.imread('GoldFish.jpg')
#img = cv2.resize(img,(250,250),interpolation=cv2.INTER_CUBIC)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()
#image = tf.constant(['GoldFish.jpg'])
#image_string = tf.read_file(image)
image_decoded = tf.image.decode_jpeg(img)
image_resized = tf.image.resize_images(image_decoded,[250,250])
print(sess.run(image_resized))
