import os
from time import time
import os.path
import errno

import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
NUM_EPOCHS = 1000
approx = True


class PolyAct(Layer):
    def __init__(self, **kwargs):
        super(PolyAct, self).__init__(**kwargs)

    def build(self, input_shape):
        self.coeff = self.add_weight('coeff', shape=(2, 1), initializer="random_normal", trainable=True, )

    def call(self, inputs):
        return self.coeff[1] * K.square(inputs) + self.coeff[0] * inputs

    def compute_output_shape(self, input_shape):
        return input_shape


if approx:
    internal_activation = None
    add_activation_layer = lambda x: PolyAct()(x)
    pool = layers.AvgPool2D
else:
    internal_activation = 'relu'
    add_activation_layer = lambda x: x
    pool = layers.MaxPool2D


# From https://github.com/toxtli/SqueezeNet-CIFAR10-keras/
def fire_module(x, s1x1, e1x1, e3x3, name):
    # Squeeze layer
    squeeze = layers.Conv2D(s1x1, (1, 1), activation=internal_activation, padding='valid',
                            kernel_initializer='glorot_uniform',
                            name=name + 's1x1')(x)
    squeeze_act = add_activation_layer(squeeze)
    squeeze_bn = layers.BatchNormalization(name=name + 'sbn')(squeeze_act)

    # Expand 1x1 layer and 3x3 layer are parallel

    # Expand 1x1 layer
    expand1x1 = layers.Conv2D(e1x1, (1, 1), activation=internal_activation, padding='valid',
                              kernel_initializer='glorot_uniform',
                              name=name + 'e1x1')(squeeze_bn)
    expand1x1_act = add_activation_layer(expand1x1)

    # Expand 3x3 layer
    expand3x3 = layers.Conv2D(e3x3, (3, 3), activation=internal_activation, padding='same',
                              kernel_initializer='glorot_uniform',
                              name=name + 'e3x3')(squeeze_bn)
    expand3x3_act = add_activation_layer(expand3x3)

    # Concatenate expand1x1 and expand 3x3 at filters
    output = layers.Concatenate(axis=3, name=name)([expand1x1_act, expand3x3_act])

    return output


# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# Taken from https://stackoverflow.com/a/23794010/2227414
def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')


def main():
    # Load CIFAR-10 dataset. These are just regular numpy arrays.
    train, test, validation = {}, {}, {}
    (train['features'], train['labels']), (test['features'], test['labels']) = tf.keras.datasets.cifar10.load_data()

    # Create Validation set
    train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(
        train['features'], train['labels'], test_size=0.2, random_state=0)

    # Define the model
    inputs = layers.Input([32, 32, 3])

    # CONV
    conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same',
                          use_bias=True, activation=internal_activation)(inputs)
    conv1act = add_activation_layer(conv1)

    # POOL
    pool1 = pool(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1act)

    # FIRE
    fire2 = fire_module(pool1, 16, 64, 64, "Fire2")

    # FIRE
    fire3 = fire_module(fire2, 16, 64, 64, "Fire3")

    # POOL
    pool2 = pool(pool_size=(3, 3), strides=(2, 2), padding='same')(fire3)

    # FIRE
    fire4 = fire_module(pool2, 16, 64, 64, "Fire4")

    # FIRE
    fire5 = fire_module(pool2, 16, 64, 64, "Fire5")

    # CONV
    conv2 = layers.Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True,
                          activation=internal_activation)(fire5)
    conv2act = add_activation_layer(conv2)

    # AVERAGE POOL (always AVG)
    avgs = layers.GlobalAveragePooling2D()(conv2act)

    # SOFTMAX
    predictions = layers.Softmax()(avgs)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    # Model Summary
    model.summary()

    # Compile Model
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='ADAM', metrics=['accuracy'])

    x_train, y_train = train['features'], utils.to_categorical(train['labels'])
    x_validation, y_validation = validation['features'], utils.to_categorical(validation['labels'])

    train_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    ).flow(x_train, y_train, batch_size=BATCH_SIZE)

    validation_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    ).flow(x_validation, y_validation, batch_size=BATCH_SIZE)

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

    for idx, layer in enumerate(model.layers):
        prefix = './model/' + "{:02d}_".format(idx) + layer.get_config()['name']
        with safe_open_w(prefix + '_config.txt') as config:
            config.write(str(layer.get_config()))
        if layer.get_weights():
            with safe_open_w(prefix + '_weights.txt') as weights:
                weights.write(str(layer.get_weights()[0]))
        if len(layer.get_weights()) > 1:
            with safe_open_w(prefix + '_biases.txt') as biases:
                biases.write(str(layer.get_weights()[1]))


if __name__ == '__main__':
    main()
