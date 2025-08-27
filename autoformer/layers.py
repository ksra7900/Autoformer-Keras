from keras.models import Model
from keras.layers import Layer, AveragePooling1D, Input, Dense

class SeriesDecomposition(Layer):
    def __init__(self, kernel_size= 25, padding= 'same', **kwargs):
        super(SeriesDecomposition, self).__init__(**kwargs)
        self.kernel_size= kernel_size
        self.padding= padding
        self.avg_pool= AveragePooling1D(
            pool_size= kernel_size,
            strides= 1,
            padding= padding
            )   
        
    def call(self, inputs):
        # Apply MovingAverage
        trend= self.avg_pool(inputs)
        
        # Seasonal Component
        seasonal= inputs - trend
        
        return seasonal, trend

    def get_config(self):
        config= super().get_config()
        config.update({
            'kernel_size' : self.kernel_size,
            'padding' : self.padding
            })
        return config

class AutoCorrelation(Layer):
    pass


if __name__ == "__main__":
    input_layer = Input(shape=(100, 5))  # (seq_len=100, features=5)
    seasonal, trend = SeriesDecomposition(kernel_size=25)(input_layer)
    
    x = Dense(64)(seasonal)
    output_layer = Dense(1)(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()