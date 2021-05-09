import tensorflow as tf
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
#sess.run(tf.global_variables_initializer())

def parser(record):
    # In process of making tfrecord file the key(here 'image_raw') should be same.
    keys_to_features = {'image_raw':tf.FixedLenFeature([],tf.string)}
                         #'label': tf.FixedLenFeature([],tf.int64)}
    parsed = tf.parse_single_example(record,keys_to_features)
    image = tf.decode_raw(parsed['image_raw'],tf.uint8) #Reinterpret the bytes of a string as a vector of numbers.
    #image = tf.cast(image,tf.float32)
    image = tf.reshape(image,shape=[224,224,3])
    #label = tf.cast(parsed['label'],tf.int32)
    return image#,label

def input_fn(filenames,train,batch_size=64,buffer_size=2048):
    #buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        #number of elements from this dataset from which the new
        #dataset will sample.
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)
    if train:
        #dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch = iterator.get_next()
    x = {'image':images_batch}
    #y = labels_batch
    return x#,y

def train_input_fn():
    return input_fn(filenames=['train1.tfrecord'],train=True)
def val_input_fn():
    return input_fn(filenames=['val.tfrecord'],train=False)
## ------------------------------
## Function to print images start
## ------------------------------

features = train_input_fn()
while True:
    try:
        img = sess.run(features['image'])#,labels)
        print(img.shape)
        for i in range(img.shape[0]):
            plt.imshow(img[i])
            plt.show()
    except tf.errors.OutOfRangeError:
        print('end of image')
        break
"""
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
"""
#print(img.shape)#,label.shape)
#plt.imshow(img[0])
#plt.show()
# Loop over each example in batch
#for i in range(img.shape[0]):
#    cv2.imshow('image',img[i])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    print('class label'+str(np.argmax(label[i])))
"""
feature_columns = [tf.feature_column.numeric_column('image',shape=[224,224,3])]

classifier = tf.estimator.DNNClassifier(
             feature_columns=feature_columns,# The input features to our model
             hidden_units=[128,64,32,16],# 5 layers
             activation_fn=tf.nn.relu,
             n_classes=3,
             model_dir='model',
             optimizer=tf.train.AdamOptimizer(1e-4),
             dropout=0.1)

classifier.train(input_fn=train_input_fn,steps=100000)
result = classifier.evaluate(input_fn=val_input_fn)
print(result)
print('Classification accuracy:{:.2f}'.format(result['accuracy']))

def model_fn(features,labels,mode,params):
    num_classes = 2
    net = features['image']
    net = tf.reshape(net,[-1,224,224,3])
    net = tf.layers.conv2d(inputs=net,name='layer_conv1',filters=32,kernal_size=3,padding='same',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net,pool_size=2,strides=2)
    net = tf.layers.conv2d(inputs=net,name='layer_conv2',filters=32,kernal_size=3,padding='same',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net,pool_size=2,strides=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net,name='layer_fc1',units=128,activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net,name='layer_fc2',units=num_classes)
    logits = net
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred,axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_cross_entropy_with_logits(lables=lables,logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        metrics = {'accuracy':tf.metrics.accuracy(labels,y_pred_cls)}
        spec = tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,eval_metric_ops=metrics)
    return spec

model = tf.estimator.Estimator(model_fn=model_fn,params={'learning_rate':1e-4},model_dir='./model2')

count = 0
while (count < 100):
    model.train(input_fn=train_input_fn,steps=1000)
    result = model.evaluate(input_fn=val_input_fn)
    print(result)
    print('Classification accuracy: {:.2f}'.format(result['accuracy']))
    sys.stdout.flush()
    count = count + 1
"""
