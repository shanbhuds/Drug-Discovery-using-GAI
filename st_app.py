import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

latent_dim = 64
input_dim = 4

# Load your dataset
train = pd.read_csv("drug_dataset.csv")
train.dropna(inplace=True)

# Assuming train_data is extracted from your dataset
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
decoder_outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder_inputs)
decoder = tf.keras.models.Model(inputs=decoder_inputs, outputs=decoder_outputs)

encoder_decoder_outputs = decoder(z)
vae = tf.keras.models.Model(encoder_inputs, encoder_decoder_outputs)
vae.compile(optimizer='adam', loss='mse')

def generate_drug(user_input):
    output = vae.predict(user_input)
    return output

def main():
    st.title('Drug Compound Generator')
    st.write('This app generates new drug compounds using Variational Autoencoder.')
    user_input = st.text_input('Enter drug compound features separated by commas (e.g., 0.2, 0.3, 0.4, 0.5):')
    
    if st.button('Generate Compound'):
        features = [float(x.strip()) for x in user_input.split(',')]
        user_input = np.array([features])
        output = generate_drug(user_input)
        st.write('Generated Drug Compound:')
        st.write(output)

if __name__ == '__main__':
    main()
