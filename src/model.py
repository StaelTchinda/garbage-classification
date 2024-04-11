
import keras
import tensorflow as tf


### FIX for the following error reated to SSL: see https://github.com/tensorflow/tensorflow/issues/33285
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
### END FIX

def get_model(class_count: int) -> keras.Model:
    base_model = keras.applications.EfficientNetV2B1(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    keras_model = keras.models.Sequential()
    keras_model.add(base_model)
    keras_model.add(keras.layers.Flatten())
    keras_model.add(keras.layers.Dropout(0.5))
    keras_model.add(keras.layers.Dense(class_count,activation=tf.nn.softmax))

    return keras_model
