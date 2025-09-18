import math
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, AveragePooling1D, Input, Dense, Dropout

# intialize Series Decomposition
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
    
# intialize AutoCorrelation
class AutoCorrelation(Layer):
    def __init__(self, d_model, n_heads, c=1, **kwargs):
        super(AutoCorrelation, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.c = c
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        
    def build(self, input_shape):
        # Define weight matrices Q,V,K for forecasting
        query_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        self.wq= self.add_weight(shape=(query_shape[-1], self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wq')
        self.wk= self.add_weight(shape=(query_shape[-1], self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wk')
        self.wv= self.add_weight(shape=(query_shape[-1], self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wv')
        # Outout forecast matrices
        self.wo= self.add_weight(shape=(self.d_model, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wo')
        
        
        super(AutoCorrelation, self).build(input_shape)
        
    def _time_delay_agg(self, values, corr):
        batch_size = tf.shape(values)[0]
        n_heads = tf.shape(values)[1]
        d_head = tf.shape(values)[2]
        length = tf.shape(values)[3]
        
        # Find top k delays
        #k= int(self.c * math.log(length))
        kـfloat= self.c * tf.math.log(tf.cast(length, tf.float32))
        k= tf.cast(kـfloat, tf.int32)
        k= tf.maximum(k, 1)
        k = tf.minimum(k, length)
        
        mean_corr= tf.reduce_mean(corr, axis=[1, 2])
        
        top_k_vals, top_k_indices= tf.math.top_k(mean_corr, k=k)
        
        # Apply softmax to get weights
        weights= tf.nn.softmax(top_k_vals, axis=-1)
        
        # time delay aggregation
        delays_agg= tf.zeros_like(values)
            
        def process_single_delay(i_and):
            i = i_and[0]  # scalar index, not used directly
            delay_idx = i_and[1]  # shape [batch]
            weight = i_and[2]     # shape [batch]
            # For each batch b, roll values[b] by -delay_idx[b] along axis=1 (seq axis inside values[b])
            def roll_for_batch(b):
                # values[b] shape [n_heads, seq_len, d_head]
                shift = -delay_idx[b]
                return tf.roll(values[b], shift=shift, axis=1)
            rolled = tf.map_fn(roll_for_batch, tf.range(batch_size), dtype=values.dtype)
            # rolled shape: [batch, n_heads, seq_len, d_head]
            weight_exp = tf.reshape(weight, [batch_size, 1, 1, 1])
            return rolled * tf.cast(weight_exp, values.dtype)
        
        elems= (tf.range(k),
                tf.transpose(top_k_indices, perm=[1, 0]),
                tf.transpose(weights, perm=[1, 0])
            )
        
        results = tf.map_fn(lambda x: process_single_delay(x),
                            elems, dtype=values.dtype)
        delays_agg = tf.reduce_sum(results, axis=0)
            
        return delays_agg
        
        
    def call(self, inputs):
        queries, keys, values = inputs
        batch_size = tf.shape(queries)[0]
        len_q = tf.shape(queries)[1]
        len_k = tf.shape(keys)[1]
        
        # Linear forecast and split into heads
        Q = tf.matmul(queries, self.wq)
        K = tf.matmul(keys, self.wk)
        V = tf.matmul(values, self.wv)
        
        # Reshape to (batch_size, seq_len, n_heads, d_head)
        Q = tf.reshape(Q, (batch_size, len_q, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch_size, len_k, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch_size, len_k, self.n_heads, self.d_head))
        
        # Transpose to (batch_size, n_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # compute FFT 
        Q_fft = tf.signal.rfft(Q)
        K_fft = tf.signal.rfft(K)
        
        # Compute cross-correlation via inverse FFT of element-wise product
        S = Q_fft * tf.math.conj(K_fft)
        corr = tf.signal.irfft(S)
        corr = tf.math.real(corr)
        
        corr = tf.einsum('b h i d, b h j d -> b h i j', Q, K)
        
        # delay aggregation
        agg_output= self._time_delay_agg(V, corr)
        
        # Transpose delay
        agg_output= tf.transpose(agg_output, perm=[0, 3, 1, 2])
        
        # Reshape to (batch_size, seq_len, d_model) 
        agg_output = tf.reshape(agg_output, (batch_size, len_q, self.d_model))
        
        # Final linear projection
        output = tf.matmul(agg_output, self.wo)
        
        return output
    
    def compute_output_shape(self, input_shape):
        query_shape = input_shape[0]
        return (query_shape[0], query_shape[1], self.d_model)

class AutoCorrelationWrapper(Layer):
    def __init__(self, d_model, n_heads, c=1, **kwargs):
        super(AutoCorrelationWrapper, self).__init__(**kwargs)
        self.auto_correlation = AutoCorrelation(
            d_model=d_model, 
            n_heads=n_heads, 
            c=c
        )
        
    def call(self, inputs):
        # inputs should be a list of [queries, keys, values]
        queries, keys, values= inputs
        return self.auto_correlation(inputs)
    
    def compute_output_shape(self, input_shape):
        return self.auto_correlation.compute_output_shape(input_shape)
    
if __name__ == "__main__":
    # input
    '''input_layer = Input(shape=(100, 5))
    d_model = 8
    n_heads = 4
    
    # creat Q, V, K
    projected = Dense(d_model)(input_layer)
    
    # AutoCorrelation (self-attention)
    ac_output = AutoCorrelationWrapper(d_model=d_model, n_heads=n_heads)(
        [projected, projected, projected]  
    )
    
    seasonal, trend = SeriesDecomposition(kernel_size=25)(ac_output)
    
    output_layer = seasonal + trend  
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()'''
    
    layer = AutoCorrelation(d_model=16, n_heads=4)
    q = tf.random.normal((2, 24, 16))
    k = tf.random.normal((2, 24, 16))
    v = tf.random.normal((2, 24, 16))
    
    out = layer([q, k, v])
    print(out.shape)