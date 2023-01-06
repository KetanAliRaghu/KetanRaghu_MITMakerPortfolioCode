"""
Deep Convolutional Generative Adversarial Network
"""

import keras
import keras_preprocessing.image
import tensorflow as tf
from keras import layers as l
from keras import optimizers , losses
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

dataset = tfds.load(name = 'celeb_a',
                    split = 'train',
                    shuffle_files = True,
                    batch_size = 32).map(lambda x: x/255)

discriminator = keras.Sequential([
    l.Input(shape = (64 , 64 , 3)),
    l.Conv2D(64 , kernel_size = 4 , strides = 2 , padding = 'same'),
    l.LeakyReLU(0.2),
    l.Conv2D(128 , kernel_size = 4 , strides = 2 , padding = 'same'),
    l.LeakyReLU(0.2),
    l.Conv2D(128 , kernel_size = 4 , strides = 2 , padding = 'same'),
    l.LeakyReLU(0.2),
    l.Flatten(),
    l.Dropout(0.2),
    l.Dense(1 , activation = 'sigmoid')
])

print(discriminator.summary())

latent_dim = 128
generator = keras.Sequential([
    l.Input(shape = (latent_dim,)),
    l.Dense(8192),
    l.Reshape(8 , 8 , 128),
    l.Conv2DTranspose(128 , kernel_size = 4 , strides = 2 , padding = 'same'),
    l.LeakyReLU(0.2),
    l.Conv2DTranspose(256 , kernel_size = 4 , strides = 2 , padding = 'same'),
    l.LeakyReLU(0.2),
    l.Conv2DTranspose(512 , kernel_size = 4 , strides = 2 , padding = 'same'),
    l.LeakyReLU(0.2),
    l.Conv2D(3 , kernal_size = 5, padding = 'same' , activation = 'sigmoid')
])

print(generator.summary())

opt_gen = optimizers.Adam(1e-4)
opt_disc = optimizers.Adam(1e-4)
loss_fn = losses.BinaryCrossentropy()

for epoch in range(10):
    for idx , real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape = (batch_size , latent_dim))
        fake = generator(random_latent_vectors)

        if idx % 100 == 0:
            img = keras_preprocessing.image.array_to_img(fake[0])
            # img.save(f'data/generated_images/generated_img{epoch}_{idx}.png')

        # D = Discriminator; G = Generator
        # Train Discriminator: max y * log(D(x)) + (1 - y) * log(1 - D(G(z))
        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size , 1)) , discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros(batch_size , 1) , discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        grads = disc_tape.gradient(loss_disc , discriminator.trainable_weights)
        opt_disc.apply_gradients(zip(grads , discriminator.trainable_weights))

        # Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size , 1) , output)


        grads = gen_tape.gradient(loss_gen , generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads , discriminator.trainable_weights))
