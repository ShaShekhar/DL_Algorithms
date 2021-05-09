import random
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) # In addition to value themself you store type of list.

def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def createDataRecord(out_filename,addrs):#,labels):
    #open the TFRecords file,A class to write records to a TFRecords file.
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i%1000:
            print('Train data: {}/{}'.format(i,len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])
        #label = labels[i]
        if img is None:
            continue
        # Create a feature
        feature = {'image_raw': _bytes_feature(img.tostring())}#,'label': _int64_feature(label)}
        #b'\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\...., This is the result of img.tostring()
        """The same key must use when parsing the file"""
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # It returns the json format data.
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

cat_dog_train_path = './image_data/*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
#labels = [0 if 'Cat' in addr else 1 for addr in addrs]

# shuffle data
#c = list(zip(addrs,labels))
random.shuffle(addrs)
#addrs,labels = zip(*c)

#Divide the data into 60% train,20% validation, and 20% test
train_addrs1 = addrs[0:int(0.1*len(addrs))]
#train_addrs2 = addrs[int(0.4*len(addrs)):int(0.8*len(addrs))]
#train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.8*len(addrs)):int(0.9*len(addrs))]
#val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.9*len(addrs)):]
#test_labels = labels[int(0.8*len(labels)):]

createDataRecord('train1.tfrecord',train_addrs1)#,train_labels)
#createDataRecord('train2.tfrecord',train_addrs2)
#createDataRecord('val.tfrecord',val_addrs)#,val_labels)
#createDataRecord('test.tfrecord',test_addrs)#,test_labels)
