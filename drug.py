# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:05:38 2024

@author: prasarad1
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


train = pd.read_csv("drug_dataset.csv")
train.dropna(inplace=True)
latent_dim = 64
input_dim = 4

train_data = np.random.rand(1000, input_dim)
val_data = np.random.rand(200, input_dim)

encoder_inputs = tf.keras.layers.Input(shape=(input_dim,))
encoder_outputs = tf.keras.layers.Dense(64, activation='relu')(encoder_inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(encoder_outputs)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(encoder_outputs)


# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

# Define the decoder
decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
# Define decoder layers
decoder_outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder_inputs)

decoder = tf.keras.models.Model(inputs=decoder_inputs, outputs=decoder_outputs)

encoder_decoder_outputs = decoder(z)

vae = tf.keras.models.Model(encoder_inputs, encoder_decoder_outputs)
vae.compile(optimizer='adam', loss='mse')


vae.fit(train_data, train_data, epochs=10, batch_size=32, validation_data=(val_data, val_data))


vae.save("your_vae_model.h5")

user_input = np.array([[0.2, 0.3, 0.4, 0.5]])
output = vae.predict(user_input)

