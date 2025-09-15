import tensorflow as tf
from keras.models import Model, Sequential
from keras import layers, Input
from layers import AutoCorrelationWrapper, SeriesDecomposition

# Encoder block
class encoder(layers.Layer):
    def __init__(self, 
                 d_model,
                 n_heads,
                 conv_filter,
                 activation='relu',
                 n=1,
                 kernel_size= 25,
                 padding= 'same',
                 dropout_rate= 0.1,
                 c=1,
                 **kwargs):
        super(encoder, self).__init__(**kwargs)
        self.n= n
        self.kernel_size= kernel_size
        self.padding= padding
        self.dropout_rate= dropout_rate
        self.c= c
        self.d_model= d_model
        self.n_heads= n_heads
        self.activation= activation
        self.conv_filter= conv_filter
        # SeriesDecomposition layer
        self.moving_avg= SeriesDecomposition(kernel_size=self.kernel_size,
                                             padding=self.padding)
        
        # fully connect layer
        self.ff_layer= None
    
    def build(self, input_shape):
        self.ff_layer= Sequential(
            [
                layers.Conv1D(filters= self.conv_filter,
                              kernel_size= 1,
                              activation=self.activation,
                              use_bias= False),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(filters= self.conv_filter,
                              kernel_size= 1,
                              activation= None,
                              use_bias= False),
                layers.Dropout(self.dropout_rate)
            ]
        )
        
    def call(self, inputs):
        x= layers.Dense(self.d_model)(inputs)
        x= AutoCorrelationWrapper(d_model=self.d_model, 
                                  n_heads=self.n_heads)([x,x,x])
        x= x + inputs
        x,_= self.moving_avg(x)
            
        y= self.ff_layer(x)
        y, _= self.moving_avg(y)
        
        return y
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class decoder(layers.Layer):
    def __init__(self,
                 d_out,
                 d_model,
                 n_heads,
                 conv_filter,
                 activation='relu',
                 n=1,
                 kernel_size= 25,
                 padding= 'same',
                 dropout_rate= 0.1,
                 c=1,
                 **kwargs):
        super(decoder, self).__init__(**kwargs)
        self.n= n
        self.kernel_size= kernel_size
        self.padding= padding
        self.dropout_rate= dropout_rate
        self.c= c
        self.d_model= d_model
        self.n_heads= n_heads
        self.activation= activation
        self.conv_filter= conv_filter
        # SeriesDecomposition layer
        self.moving_avg= SeriesDecomposition(kernel_size=self.kernel_size,
                                             padding=self.padding)
        
        # fully connect layer
        self.ff_layer= None
        
        # output layer
        self.out_proj = layers.Conv1D(filters=d_out, kernel_size=3,
                                 strides=1, padding="same",
                                 use_bias=False)
        
    def build(self, input_shape):
        self.ff_layer= Sequential(
            [
                layers.Conv1D(filters= self.conv_filter,
                              kernel_size= 1,
                              activation=self.activation,
                              use_bias= False),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(filters= self.conv_filter,
                              kernel_size= 1,
                              activation= None,
                              use_bias= False),
                layers.Dropout(self.dropout_rate)
            ]
        )
        
    def call(self, inputs:tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        x, cross= inputs
        x += AutoCorrelationWrapper(d_model=self.d_model, 
                                  n_heads=self.n_heads)([x,x,x])
        x, xt_1= self.moving_avg(x)
        
        x += AutoCorrelationWrapper(d_model=self.d_model, 
                                  n_heads=self.n_heads)([x, cross, cross])
        x, xt_2= self.moving_avg(x)
        
        y = self.out_proj(x)
        y, xt_3= self.moving_avg(x + y)
        
        residualt_trend= self.out_proj(xt_1, xt_2, xt_3)
        output= residualt_trend + y
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
if __name__ == '__main__':
    
    input_layer = Input(shape=(100, 5))
    d_model = 8
    n_heads = 4
    x = encoder(d_model=d_model,
                n_heads=n_heads,
                conv_filter=16)(input_layer)
    
    x = decoder(d_model=d_model,
                n_heads=n_heads,
                conv_filter=16,
                d_out=16)(x)
    
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    

