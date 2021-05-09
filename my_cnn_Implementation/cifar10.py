import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
sess = tf.Session()
# declare model parameter
batch_size = 128
output_every = 50
iteration = 20000
eval_every = 500
image_height = 32
image_width = 32
crop_height = 24
crop_width = 24
num_channels = 3
num_target = 10
data_dir = 'temp1'
extract_folder = 'cifar-10-batches-bin'
# the initial lr will be set at 0.1,and we will exponentially decrease it by a factor of 10% every 250 iterations.
# TF does accept a 'staircase' argument which only updates the lr.
learning_rate = 0.1
lr_decay = 0.9
num_iter_to_wait = 250
# Now we'll set up parameters so that we can read in the binary CIFAR-10 images:
image_vec_length = image_height*image_width*num_channels
record_length = 1 + image_vec_length
# Next, we'll set up the data directory and the URL to download the CIFAR-10 images.
data_dir = 'temp1'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
data_file = os.path.join(data_dir,'cifar-10-binary.tar.gz')
if not os.path.isfile(data_file):
    # download file
    filepath,_ = urllib.request.urlretrieve(cifar10_url, data_file)#, progress)
    tarfile.open(filepath,'r:gz').extractall(data_dir)

# we will set up the record reader and return a randomly distorted image with the following 'read_cifar_file()' function.
# First,we need to declare a record reader object that will read in a fixed length of bytes.After we read the image queue,
# we'll split apart the image and label.Finally, we will randomly distort the image with TF's built in image modification functions:
def read_cifar_files(filename_queue,distort_images=True):
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key,record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string,tf.uint8)
    # Extract label
    image_label = tf.cast(tf.slice(record_bytes,[0],[1]),tf.int32)
    # Extract image
    image_extracted = tf.reshape(tf.slice(record_bytes,[1],[image_vec_length]), [num_channels,image_height,image_width])
    # Reshape image
    image_uint8image = tf.transpose(image_extracted, [1,2,0])
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # Randomly Crop image
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,crop_width,crop_height)
    if distort_images:
        # Randomly flip the image horizontally, chane the brightness and contrast
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image,max_delta=63)
        final_image = tf.image.random_contrast(final_image,lower=0.2,upper=1.8)
        # Normalize whitening
        final_image = tf.image.per_image_whitening(final_image)
    return (final_image,image_label)
# Now we'll declare a function that will populate our image pipeline for batch processor to use.We first need to set up the file
# list of images we want to read through,and to define how to read them with an input producer object,created through prebuilt
# TF functions.The input producer can be passed into the reading function that we created in the preceding step,'resd_cifar_files()'.
# We'll then set a batch reader on the queue,shuffle_batch():
def input_pipeline(batch_size,train_logical=True):
    if train_logical:
        files = [os.path.join(data_dir,extract_folder,'data_batch_{}.bin'.format(i)) for i in range(1,6)]
    else:
        files = [os.path.join(data_dir,extract_folder,'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image,label = read_cifar_files(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3*batch_size
    example_batch,label_batch = tf.train.shuffle_batch([imageimport pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(files):
    with open(files, 'rb') as f:
        dict1 = pickle.load(f,encoding='latin1')
    return dict1

label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

def load_data_wrapper():
    data = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_1').get('data')
    label = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_1').get('labels')
    training_data = [np.transpose(np.reshape(x,(3,32,32)),(1,2,0)) for x in data]
    training_label = [label_dict[x] for x in label]
    for i in range(100):
        ax = plt.subplot(10,10,i+1)
        ax.set_title(training_label[i])
        plt.imshow(training_data[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
load_data_wrapper()
,label],batch_capacity,capacity,min_after_dequeue)
    return (example_batch,label_batch)
# Next,we can declare our model function.The model we will use has two convolutional layers,followed by three fully connected layers.
# To make variable declaration easier,we'll start by declaring two variable functions.The two convolutional layers will create 64
# features each.The first fully connected layer will connect 2nd convolutional layer with 384 hidden nodes.
# The second fully connected operation will connect those 384 hidden nodes to 192 hidden nodes.The final hidden layer operation
#  will then connect the 192 nodes to the 10 output classes we are trying to predict.
def cifar_cnn_model(input_images,batch_size,train_logical=True):
    def truncated_normal_var(name,shape_dtype):
        return (tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.truncated_normal_initializer(stddev=0.05)))
    def zero_var(name,shape,dtype):
        return (tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.constant_initializer(0.0)))
    # First Convolutional Layer
    with tf.variable_scope('conv1') as scope:
        # conv_kernal is 5*5 for all 3 colors and we will create 64 features
        conv1_kernal = truncated_normal_var(name='conv_kernal1',shape=[5,5,3,64],dtype=tf.float32)
        # We convolve across the in=mage with a stride size of 1
        conv1 = tf.nn.conv2d(input_images,conv1_kernal,[1,1,1,1],padding='SAME')
        # Initialize and add bias term
        conv1_bias = zero_var(name='conv_bias1',shape=[64],dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1,conv1_bias)
        # ReLU element wise
        relu_conv1 = tf.nn.relu(conv1_add_bias)
    # Max Pooling
    pool1 = tf.nn.max_pool(relu_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool_layer1')
    # Local Response Normalization
    norm1 = tf.nn.lrn(pool1,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm1')
    # Second Convolutional Layer
    with tf.variable_scope('conv2') as scope:
        conv2_kernal = truncated_normal_var(name='conv2_kernal',shape=[5,5,64,64],dtype=tf.float32)
        conv2 = tf.nn.conv2d(norm1,conv2_kernal,[1,1,1,1],padding='SAME')
        # Initilize and add the bias
        conv2_bias = zero_var(name='conv2_bias',shape=[64],dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2,conv2_bias)
        # ReLU element wise
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    # Max Pooling
    pool2 = tf.nn.max_pool(relu_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool_layer2')
    # Local Response Normalization
    norm2 = tf.nn.lrn(pool2,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm2')
    # Reshape output into a single matrix for multiplication for fully connected layers
    reshaped_output = tf.reshape(norm2,[batch_size,-1])
    reshape_dim = reshaped_output.get_shape()[1].value

    # First Fully Connected Layer
    with tf.variable_scope('full1') as scope:
        full_weight1 = truncated_normal_var(name='full_mult1',shape=[reshaped_dim,384],dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1',shape=[384],dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight1),full_bias1))
    # Second Fully Connected Layer
    with tf.variable_scope('full2') as scope:
        full_weight2 = truncated_normal_var(name='full_mult2',shape=[384,192],dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2',shape=[192],dtype=tf.flaot32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1,full_weight2),full_bias2))
    # Third Fully Connected Layer
    with tf.variable_scope('full3') as scope:
        full_weight3 = trunacted_normal_var(name='full_mult3',shape=[192,10],dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2',shape=[10],dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2,full_weight3),full_bias3)
        return (final_output)
# Now we'll create the 'loss' function.We will use the 'softmax' function because a picture can only take on
# exactly one category,so the output should be a probability distribution over rhe ten targets:
def cifar_loss(logits,targets):
    # Get rid of extra dimensions and cast target into integers
    targets = tf.squeeze(tf.cast(targets,tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=targets)
    # Take the average loss across batch size
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return (cross_entropy_mean)
# Now we declare our training step.The lr will decrease in an exponential step function:
def train_step(loss_value,iteration_num):
    model_learning_rate = tf.train.exponential_decay(learning_rate,iteration_num,num_iter_to_wait,lr_decay,staircase=True)
    # create optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return (train_step)

def accuracy_of_batch(logits,targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(taregts,tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_prediction = tf.cast(tf.argmax(logits,1),tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions,targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly,tf.float32))
    return (accuracy)
# Now we have an 'imagepipeline' function,we can initialize both the training image pipeline and the test image pipeline:
images,targets = input_pipeline(batch_size,train_logical=True)
test_images,test_targets = input_pipeline(batch_size,train_logical=False)

# Next,we'll initialize the model for the training output and the test output.It is important to note that we must declare
# 'scope.reuse_variables()' after we create the training model so that,when we declare the model for the test network,
# it will use the same model parameters:
with tf.variable_scope('model_definition') as scope:
    # Declare the training network model
    model_output = cifar_cnn_model(images,batch_size)
    # Use same varibles within scope
    scope.reuse_variables()
    # Declare test model output
    test_output = cifar_cnn_model(test_images,batch_size)
# We can now initialize our loss and test accuracy functions.Then we'll declare the iteration variable.This variable needs to be declared
# as non-trainable,and passed to our training function that uses it in the learning rate exponential decay calculation:
loss = cifar_loss(model_output,targets)
accuracy = accuracy_of_batch(test_output,test_targets)
iteration_num = tf.Variable(0,trainable=False)
train_op = train_step(loss,iteration_num)
# We'll now initialize all of the model's variables and then start the image pipeline by running the TF function,'start_queue_runners()'.
# When we start the 'train' or 'test' model output,the pipeline will feed in a batch of images in place of a feed dictionary:
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
# We now loop through our training iteration and save the training loss and the test accuracy
train_loss = []
test_accuracy = []
for i in range(iteration):
    _,loss_value = sess.run([train_op,loss])
    if (i+1) % output_every == 0:
        train_loss.append(loss_value)
        output = 'Iteration {}:Loss = {:.5f}'.format((i+1),loss_value)
        print(output)
    if (i+1)%eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = '--- Test Accuracy={:.2f}%.'.format(100.*temp_accuracy)
        print(acc_output)

eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)
# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Softmax Loss')
plt.show()
# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
