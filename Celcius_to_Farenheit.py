'''
Following an intro tutorial from Colab:
https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb#scrollTo=YHI3vyhv5p85
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

#Data
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

print('+++++++++++++++++++++++++++++++++++++++++')
for x,y in enumerate(celsius_q):
    print("{} Celcius is equal to {} Farenheit".format(y, fahrenheit_a[x]))
print('+++++++++++++++++++++++++++++++++++++++++')

#Create the model
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])]
)

#Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

#train the model
history = model.fit(celsius_q,fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

#plot the results
plt.title('Loss vs Epoch')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show()

#predict farenheit by passing a celcius value
print(model.predict([100.00]))




