import tensorflow as tf
from keras.models import Sequential
from keras import layers
from layers import AutoCorrelationWrapper, SeriesDecomposition

# Encoder block
class encoder(layers.Layer):
    def __init__(self, 
                 d_model,
                 n_heads,
                 conv_filter,
                 activation='relu',
                 kernel_size= 25,
                 padding= 'same',
                 dropout_rate= 0.1,
                 c=1,
                 **kwargs):
        super(encoder, self).__init__(**kwargs)
        self.kernel_size= kernel_size
        self.padding= padding
        self.dropout_rate= dropout_rate
        self.c= c
        self.d_model= d_model
        self.n_heads= n_heads
        self.activation= activation
        self.conv_filter= conv_filter
        # layers
        self.input_proj = layers.Dense(self.d_model)
        self.residual_proj = layers.Dense(self.d_model)
        # Series Decomposition
        self.moving_avg= SeriesDecomposition(kernel_size=self.kernel_size,
                                             padding=self.padding)
        
        # fully connect layer
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
        
        # Autocorrelation
        self.AutoCorr= AutoCorrelationWrapper(d_model=self.d_model, 
                                              n_heads=self.n_heads)
        
        
    def call(self, inputs):
        proj_input= self.residual_proj(inputs)
        x= self.input_proj(proj_input)
        x= self.AutoCorr([x,x,x])
        x= x + proj_input
        x,_= self.moving_avg(x)
            
        y= self.ff_layer(x)
        y, _= self.moving_avg(y)
        
        return y
    
    def get_config(self):
        config= super().get_config()
        config.update({
            'd_model' : self.d_model,
            'n_heads' : self.n_heads,
            'conv_filter' : self.conv_filter,
            'activation' : self.activation,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'dropout_rate' : self.dropout_rate,
            'c' : self.c,
            })
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class decoder(layers.Layer):
    def __init__(self,
                 d_out,
                 d_model,
                 n_heads,
                 conv_filter,
                 activation='relu',
                 kernel_size= 25,
                 padding= 'same',
                 dropout_rate= 0.1,
                 c=1,
                 **kwargs):
        super(decoder, self).__init__(**kwargs)
        self.kernel_size= kernel_size
        self.padding= padding
        self.dropout_rate= dropout_rate
        self.c= c
        self.d_model= d_model
        self.d_out= d_out
        self.n_heads= n_heads
        self.activation= activation
        self.conv_filter= conv_filter
        # SeriesDecomposition layer
        self.moving_avg= SeriesDecomposition(kernel_size=self.kernel_size,
                                             padding=self.padding)
        
        # fully connect layer
        self.ff_layer= None
        
        # output layer
        self.out_proj = layers.Conv1D(filters=self.d_out, kernel_size=3,
                                 strides=1, padding="same",
                                 use_bias=False)
        
    def build(self, input_shape):
        super().build(input_shape)
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
        self.AutoCorr= AutoCorrelationWrapper(d_model=self.d_model, 
                                              n_heads=self.n_heads)
        
    def call(self, inputs:tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        x, cross= inputs
        
        # self attention
        x_attn= self.AutoCorr([x,x,x])
        
        x= x + x_attn
        x_sesonal, x_trend1= self.moving_avg(x)
        
        # crosss atention
        x_cross= self.AutoCorr([cross,cross,x])
    
        x= x_sesonal + x_cross
        x_sesonal, x_trend2= self.moving_avg(x)
        
        # feed forward
        y= self.ff_layer(x_sesonal)
        y_seasonal, y_trend= self.moving_avg(y)
        
        # combined trend
        combined_trend= x_trend1 + x_trend2 + y_trend
        output_trend= self.out_proj(combined_trend)
        
        return y_seasonal + output_trend
    
    def get_config(self):
        config= super().get_config()
        config.update({
            'd_out' : self.d_out,
            'd_model' : self.d_model,
            'n_heads' : self.n_heads,
            'conv_filter' : self.conv_filter,
            'activation' : self.activation,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'dropout_rate' : self.dropout_rate,
            'c' : self.c,
            })
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape