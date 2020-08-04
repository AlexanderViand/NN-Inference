import gzip
import os
from time import time

from six.moves import urllib
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
BATCH_SIZE = 64
NUM_EPOCHS = 10


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def read_mnist(images_path: str, labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
            .reshape(length, 784) \
            .reshape(length, 28, 28, 1)
        # Normalize Data to [-0.5, 0.5]
        features = features / 255 - 0.5

    return features, labels


class PolyAct(Layer):
    def __init__(self, **kwargs):
        super(PolyAct, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight('a', shape=(1, 1), initializer="random_normal", trainable=True, )
        self.b = self.add_weight('b', shape=(1, 1), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return self.a * inputs * inputs + self.b * inputs

    def get_config(self):
        config = {'a': float(self.a), 'b': float(self.b)}
        base_config = super(Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def main():
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract the data
    train, test, validation = {}, {}, {}
    train['features'], train['labels'] = read_mnist('data/train-images-idx3-ubyte.gz',
                                                    'data/train-labels-idx1-ubyte.gz')
    test['features'], test['labels'] = read_mnist('data/t10k-images-idx3-ubyte.gz', 'data/t10k-labels-idx1-ubyte.gz')
    train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(
        train['features'], train['labels'], test_size=0.2, random_state=0)

    # Define the model
    model = tf.keras.Sequential()

    approx = True

    if approx:
        internal_activation = None
        add_activation_layer = lambda x: x.add(PolyAct())
        pool = layers.AvgPool2D
    else:
        internal_activation = 'relu'
        add_activation_layer = lambda _: None
        pool = layers.MaxPool2D

    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same',
                            use_bias=True, activation=internal_activation))
    add_activation_layer(model)
    model.add(pool(pool_size=(2, 2), padding='same'))

    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', use_bias=True, activation=internal_activation))
    add_activation_layer(model)
    model.add(pool(pool_size=(2, 2), padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=512, use_bias=True, activation=internal_activation))
    add_activation_layer(model)

    model.add(layers.Dropout(rate=0.5))

    # TODO: How did EVA/CHET handle the final softmax?
    model.add(layers.Dense(units=10, activation='softmax', use_bias=True))

    # Model Summary
    model.summary()

    # Compile Model
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='ADAM', metrics=['accuracy'])

    x_train, y_train = train['features'], utils.to_categorical(train['labels'])
    x_validation, y_validation = validation['features'], utils.to_categorical(validation['labels'])

    train_generator = ImageDataGenerator().flow(x_train, y_train, batch_size=BATCH_SIZE)
    validation_generator = ImageDataGenerator().flow(x_validation, y_validation, batch_size=BATCH_SIZE)

    print('# of training images:', train['features'].shape[0])
    print('# of validation images:', validation['features'].shape[0])

    steps_per_epoch = x_train.shape[0] // BATCH_SIZE
    validation_steps = x_validation.shape[0] // BATCH_SIZE

    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,
                        validation_data=validation_generator, validation_steps=validation_steps,
                        shuffle=True, callbacks=[TensorBoard(log_dir="logs\\{}".format(time()))])

    score = model.evaluate(test['features'], utils.to_categorical(test['labels']))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
