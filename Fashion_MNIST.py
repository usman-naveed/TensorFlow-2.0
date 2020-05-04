'''
Classifying Fashion MNIST Data using a NN with dense layers vs a CNN
'''

import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import math

tfds.disable_progress_bar()
import numpy as np
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# import the dataset
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# split into train and test
train_data, test_data = data['train'], data['test']
names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(metadata)
# explore the data
num_train = metadata.splits['train'].num_examples
num_test = metadata.splits['test'].num_examples
print("Train dataset has {} examples. \nTest dataset has {} examples".format(num_train, num_test))


# pre-process data
def normalise(images, labels):
    images = tf.cast(images, tf.float32)
    images = images / 255
    return images, labels


train_data = train_data.map(normalise)
test_data = test_data.map(normalise)

#cache the normalised data to improve efficiency
train_data = train_data.cache()
test_data = test_data.cache()

#explore the pre-processed data
for image, label in test_data.take(1):
    break
image = image.numpy().reshape((28,28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#show first 25 images
plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_data.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(names[label])
    i += 1
plt.show()

#build the dense NN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])

#build the CNN
CNN = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])

#compile the dense NN model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#compile the CNN model
CNN.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#train the dense NN model
BATCH_SIZE = 32
train_data = train_data.repeat().shuffle(num_train).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)
model.fit(train_data, epochs=5, steps_per_epoch=math.ceil(num_train/BATCH_SIZE))

#train the CNN model
CNN.fit(train_data, epochs=5, steps_per_epoch=math.ceil(num_train/BATCH_SIZE))

#evaluate the dense NN model
test_loss, test_accuracy = model.evaluate(test_data, steps=math.ceil(num_test/BATCH_SIZE))
print('Accuracy of the model: {}\n'.format(test_accuracy))

#evaluate the  CNN model
test_loss_CNN, test_accuracy_CNN = CNN.evaluate(test_data, steps=math.ceil(num_test/BATCH_SIZE))
print('Accuracy of the model (CNN): {}\n'.format(test_accuracy_CNN))

#make predictions, exploring the dense NN model
for test_images, test_labels in test_data.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions_NN = model.predict(test_images)
    predictions_CNN = CNN.predict(test_images)

print('Probability distribution for the first image (NN): ', predictions_NN[0])
print('Label for the prediction: ', np.argmax(predictions_NN[0]))
print('Actual label: {} ({})'.format(test_labels[0], names[0]))

print('Probability distribution for the first image (CNN): ', predictions_CNN[0])
print('Label for the prediction: ', np.argmax(predictions_CNN[0]))
print('Actual label: {} ({})'.format(test_labels[0], names[0]))


