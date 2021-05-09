import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28)/255.0
test_images = test_images[:1000].reshape(-1, 28*28)/255.0

def create_model():
    model = tf.keras.model.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
            keras.layers.Dense(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model

# Crate a basic model instance
model = create_model()
print(model.summary())

# The primary use case is to automatically save checkpoints during and at the end of training.This way you can use a trained model without having
# to retrain it, or pick-up training where you left of - in case the training process was interrupted.
# tf.keras.callbacks.ModelCheckpoint is a callback that performs this task.The callback takes a couple of arguments to configure checkpointing.

#checkpoint_path = 'training_1/cp.ckpt'
checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, varbose=1, period=5)
model.fit(train_images, train_labels, epochs=10, validation_data = (test_images,test_labels), callbacks=[cp_callback])

model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print('Untrained model, accuracy: {:5.2f}%'.format(100*acc))
# Then load the weights from the checkpoint, and re-evaluate.
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
loss,acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
