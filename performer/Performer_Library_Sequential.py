import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding
from tensorflow.keras.layers import Layer, Dropout, Activation, Lambda, Add
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, Reshape, Concatenate

from tensorflow.keras.utils import get_custom_objects

from performer.linear_attention import Performer


def position_encoding(x):
    num_rows, num_cols, d_model = x.get_shape().as_list()[-3:]
    ps = np.zeros([num_rows,num_cols,2],dtype=K.floatx())
    for ty in range(num_rows):
        for tx in range(num_cols):
            ps[ty,tx,:] = [(2/(num_rows-1))*ty - 1,(2/(num_cols-1))*tx - 1]
        
    ps_expand = K.expand_dims(K.constant(ps),axis=0)
    ps_tiled = K.tile(ps_expand,[K.shape(x)[0],1,1,1])
    x = K.concatenate([x,ps_tiled],axis=3)
    return x

class TransformerBlock(Layer):
    def __init__(self, num_heads, embed_dim, method, supports, d_rate=0.1, **kwargs):
        """A layer combining together the pieces to assemble
           a complete section of the Performer"""
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.method = method
        self.supports = supports
        self.att = Performer(num_heads=num_heads, key_dim=embed_dim,
                             attention_method=method, supports=supports)
        self.ffn = keras.Sequential(
            [layers.Dense(self.embed_dim, activation="relu"), layers.Dense(self.embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(d_rate)
        self.dropout2 = layers.Dropout(d_rate)

    def call(self, inputs):
        attn_output = self.att([inputs, inputs])
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'method': self.method,
            'supports': self.supports,
            })
        return config

def build_semi_performer(cc_layers, ff_layers, num_heads, embed_dim, method, supports, d_rate, transformer_depth, input_shape, num_actions):
    """"
    This function builds a convolutional + performer network:
!
    :param: cc_layers: [[num_filters, kernel_size,strides],...]
    :param: ff_layers: [[neurons, output_dropout_rate],...]
    :param: input_shape: The shape of the input - i.e. num channels and image height,width
    :param: num_actions: The number of output activations
    :return: model: The Keras model
    """
    transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, method=method, supports=supports, d_rate=d_rate)
    
    model = Sequential()
    
    model.add(Conv2D(filters=cc_layers[0][0], 
                     kernel_size=cc_layers[0][1], 
                     strides=cc_layers[0][2], 
                     input_shape=input_shape,
                     data_format='channels_first'))
    model.add(Activation('relu'))
    
    model.add(Reshape((5,5,64)))
    model.add(Dense(22))
    
    # POSITION EMBEDDING
    model.add(Lambda(position_encoding)) 
    model.add(Reshape((25,24))) 
    
    # TRANSFORMER
    for i in range(transformer_depth):
        model.add(transformer_block)

    model.add(Flatten())        
    
    # FEED-FORWARD
    for j in range(len(ff_layers)):
        model.add(Dense(ff_layers[j][0]))
        model.add(Activation('relu'))
        model.add(Dropout(rate=ff_layers[j][1]))

    model.add(Dense(num_actions))
    model.add(Activation('linear'))
    
    return model

get_custom_objects().update({
    'TransformerBlock': TransformerBlock,
})