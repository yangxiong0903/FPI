from keras.models import Model
from keras import *
import keras
from keras.layers import *
# from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
# import torch
from pyts.datasets import load_basic_motions
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize,MinMaxScaler,Binarizer
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, Conv2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import *
import tensorflow_addons as tfa
from keras.layers import Layer
from keras import regularizers
import keras.backend as K

random_state= 123
n_class=2

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

## MLSTM FCN 
def MLSTM_FCN_model(input_shape):
    n = input_shape[0]
    k = input_shape[1]
    ip = Input(shape=(n,k))

    x = LSTM(100)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(n_class, activation='softmax')(x)

    model = Model(ip, out)
    model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["AUC"]
        ) 
    return model  

def MALSTM_FCN_model(input_shape):
    n = input_shape[0]
    k = input_shape[1]
    ip = Input(shape=(n,k))

    x = Masking()(ip)
    x = LSTM(64, return_sequences=True)(x)
    x = attention()(x)
    # AttentionLSTM(64)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(2, activation='softmax')(x)

    model = Model(ip, out)
    model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["AUC"]
        ) 
    return model   

def xcm_model(input_shape):
    n = input_shape[0]
    k = input_shape[1]
    filters_num=64
    window_size=0.2
    input_layer = Input(shape=(n, k, 1))

    # 2D convolution layers
    a = Conv2D(
        filters=int(filters_num),
        kernel_size=(int(window_size * n), 1),
        strides=(1, 1),
        padding="same",
        input_shape=(n, k, 1),
        name="2D",
    )(input_layer)
    a = BatchNormalization()(a)
    a = Activation("relu", name="2D_Activation")(a)
    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name="2D_Reduced")(a)
    a = Activation("relu", name="2D_Reduced_Activation")(a)
    x = Reshape((n, k))(a)

    # 1D convolution layers
    b = Reshape((n, k))(input_layer)
    b = Conv1D(
        filters=int(filters_num),
        kernel_size=int(window_size * n),
        strides=1,
        padding="same",
        name="1D",
    )(b)
    b = BatchNormalization()(b)
    b = Activation("relu", name="1D_Activation")(b)
    b = Conv1D(filters=1, kernel_size=1, strides=1, name="1D_Reduced")(b)
    y = Activation("relu", name="1D_Reduced_Activation")(b)
    # print(x.shape,y.shape)

    # Concatenation
    z = concatenate([x, y])
    # print(z.shape)
    # 1D convolution layer
    z = Conv1D(
        filters=filters_num,
        kernel_size=int(window_size * n),
        strides=1,
        padding="same",
        name="1D_Final",
    )(z)
    z = BatchNormalization()(z)
    z = Activation("relu", name="1D_Final_Activation")(z)

    # 1D global average pooling and classification
    z = GlobalAveragePooling1D()(z)
    output_layer = Dense(n_class, activation="softmax")(z)

    model = Model(input_layer, output_layer)
    model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["AUC"]
        ) 
    return model  

def our_model(input_shape):
    n = input_shape[0]
    k = input_shape[1]
    input_layer = Input(shape=(n, k, 1))

    # 2D convolution layers
    print(input_layer.shape,'000')
    a = Conv2D(
        filters=int(64),
        kernel_size=(int(0.2 * n), 1),
        strides=(1, 1),
        padding="same",
        input_shape=(n, k, 1),
        name="2D",
    )(input_layer)
    print(a.shape, '001')
    # a = MaxPooling2D(1,1)(a)
    a = BatchNormalization()(a)
    a = Activation("relu", name="2D_Activation")(a)
    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(a)
    a = Activation("relu", name="2D_Reduced_Activation")(a) 
    print(a.shape,'100')
    x = Reshape((n, k))(a)
    print(x.shape, '101')

    # 1D convolution layers
    b = Reshape((n,k))(input_layer)
    print(b.shape,'110')
    # b = squeeze_excite_block(b)
    b =  LSTM(100, return_sequences=True)(b)
    print(b.shape,'111')
    # b = squeeze_excite_block(b)
    b =  LSTM(1, return_sequences=True)(b)
    # b = squeeze_excite_block(b)
    b = BatchNormalization()(b)
    y = Activation("relu", name="1D_Activation")(b)
    print(b.shape,'112')
    print(y.shape,'113')
    # print(x.shape,y.shape)

    # Concatenation
    z = concatenate([x, y])
    print(z.shape,'200')
    # 1D convolution layer
    z = Reshape((k+1,n,1))(z)
    z=ConvLSTM1D(
        filters=1,
        kernel_size=int(0.2 * n),
        strides=1,
        padding="same",
        name="1D",
        # dilation_rate= 1.2
    )(z)
    z = squeeze_excite_block(z)
    z = BatchNormalization()(z)
    z = Activation("relu", name="1D_Final_Activation")(z)
    print(z.shape,'201')

    # 1D global average pooling and classification
    z = GlobalAveragePooling1D()(z)
    print(z.shape,'202')
    output_layer = Dense(n_class, activation="softmax")(z)
    print(output_layer.shape,'300')

    model = Model(input_layer, output_layer)
    # model.compile(
    #         optimizer="adam", loss="categorical_crossentropy", metrics=["AUC"]
    #     ) 
    model.compile(loss= tfa.losses.SigmoidFocalCrossEntropy(), metrics=["AUC"], optimizer='sgd')
    return model 

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, LSTM, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import numpy as np

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.out_dense = Dense(embed_dim)

    def call(self, inputs):
        Q, K, V = inputs
        batch_size = tf.shape(Q)[0]
        seq_len_Q = tf.shape(Q)[1]
        seq_len_K = tf.shape(K)[1]
        
        # 计算 Q, K, V
        Q = self.query_dense(Q)
        K = self.key_dense(K)
        V = self.value_dense(V)
        
        # 分成多个头
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # 计算注意力分数
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.embed_dim // self.num_heads, tf.float32))
        
        # 计算注意力权重
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # 计算注意力输出
        context = tf.matmul(attention_weights, V)
        context = self.combine_heads(context, batch_size)
        
        # 通过最终的线性层
        out = self.out_dense(context)
        
        return out
    
    def split_heads(self, x, batch_size):
        depth = self.embed_dim // self.num_heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x, batch_size):
        depth = self.embed_dim // self.num_heads
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.embed_dim))
        return x

def stcan_model(input_shape, n_class, embed_dim=512, num_heads=8):
    n = input_shape[0]
    # print(n,1111)
    k = input_shape[1]
    # print(k,1112)
    input_layer = Input(shape=(n, k, 1))

    # 2D convolution layers
    a = Conv2D(
        filters=64,
        kernel_size=(int(0.2 * n), 1),
        strides=(1, 1),
        padding="same",
        input_shape=(n, k, 1),
        name="2D",
    )(input_layer)
    a = BatchNormalization()(a)
    a = Activation("relu", name="2D_Activation")(a)
    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(a)
    a = Activation("relu", name="2D_Reduced_Activation")(a) 
    x = Reshape((n, k))(a)

    # 1D convolution layers
    b = Reshape((n, k))(input_layer)
    b = LSTM(100, return_sequences=True)(b)
    b = LSTM(1, return_sequences=True)(b)
    b = BatchNormalization()(b)
    y = Activation("relu", name="1D_Activation")(b)


    # Cross-Attention
    z = CrossAttention(embed_dim, num_heads)([x, y, y])
    z = GlobalAveragePooling1D()(z)
    output_layer = Dense(n_class, activation="softmax")(z)

    model = Model(input_layer, output_layer)
    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=["AUC"], optimizer='sgd')
    return model


