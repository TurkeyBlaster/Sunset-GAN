from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import constraints, regularizers, initializers
from tensorflow.keras import backend as K

from gcsfs import GCSFileSystem

import numpy as np

class InstanceNormalization(Layer):
    
    def __init__(
        self,
        axis=None,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(
                shape = shape,
                name = 'gamma',
                initializer = self.gamma_initializer,
                regularizer = self.gamma_regularizer,
                constraint = self.gamma_constraint
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                shape = shape,
                name = 'beta',
                initializer = self.beta_initializer,
                regularizer = self.beta_regularizer,
                constraint = self.beta_constraint
            )
        else:
            self.beta = None

        self.built = True

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Swish(Layer):
    '''Swish activation function'''

    def __init__(self, beta=1.0, trainable=False, **kwargs):

        super().__init__(**kwargs) # Initialize super class

        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):

        self.scaling_factor = K.variable(

            self.beta,
            dtype = K.floatx(),
            name = 'scaling_factor'
        )

        if self.trainable:
            self._trainable_weights.append(self.scaling_factor)

        super().build(input_shape)

    def call(self, inputs):
        return inputs * K.sigmoid(self.scaling_factor * inputs)

    def get_config(self):

        swish_config = {

            'beta': self.get_weights()[0] if self.trainable else self.beta,
            'trainable': self.trainable
        }

        super_config = super().get_config()
        super_config.update(swish_config)

        return super_config

    def compute_output_shape(self):
        return self.input_shape

def shuffle_unison(a, b):
    seed = np.random.randint(0, 2**31 - 1)
    state = np.random.RandomState(seed)
    state.shuffle(a)
    state = np.random.RandomState(seed)
    state.shuffle(b)

def load_npz(path, project_name=None, key=None):
    
    if path[:5] == 'gs://':
        
        if project_name is None:
            fs = GCSFileSystem(token=key)
        else:
            fs = GCSFileSystem(project_name, token=key)
        file = fs.open(path)
    
    else:
        file = path
    
    print(f'Loading file {path.rsplit("/", 1)[-1]}')
    with np.load(file, allow_pickle=True) as npz:
        print(f'Available files: {npz.files}')
        X = npz[npz.files[0]]
        X = np.expand_dims(X, -1)[0]['sunset_ims']
    
    return X