import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(x, filters, stride=1):
    """2D ResNet block with skip connection"""
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection — dimension мismatch байвал 1×1 conv ашиглана
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet(input_shape, num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)  # e.g. (128, 128, 1)
    
    # Stem
    x = layers.Conv2D(32, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual stages
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return Model(inputs, outputs, name='ResNet2D')