import math
import tensorflow as tf
from keras.layers import Layer, AveragePooling1D

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
        self.max_k= None
        
    def build(self, input_shape):
        # Define weight matrices Q,V,K for forecasting
        q_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        in_dim= q_shape[-1]
        self.wq= self.add_weight(shape=(in_dim, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wq')
        self.wk= self.add_weight(shape=(in_dim, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wk')
        self.wv= self.add_weight(shape=(in_dim, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wv')
        # Outout forecast matrices
        self.wo= self.add_weight(shape=(in_dim, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wo')
        
        
        seq_len= q_shape[1]
        if isinstance(seq_len, int) and seq_len is not None:
            k= max(1, int(self.c * math.log(max(2, seq_len))))
            self.max_k= k
            
        else:
            self.max_k= 1
        
        super().build(input_shape)
        
    def call(self, inputs):
        
        queries, keys, values = inputs
        batch_size = tf.shape(queries)[0]
        len_q = tf.shape(queries)[1]
        len_k = tf.shape(keys)[1]
        
        # Linear forecast and split into heads
        Q = tf.tensordot(queries, self.wq, axes=1)
        K = tf.tensordot(keys, self.wk, axes=1)
        V = tf.tensordot(values, self.wv, axes=1)
        
        # Reshape to (batch_size, seq_len, n_heads, d_head)
        Q = tf.reshape(Q, (batch_size, len_q, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch_size, len_k, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch_size, len_k, self.n_heads, self.d_head))
        
        # Transpose to (batch_size, n_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # compute correlation
        corr = tf.einsum('b h i d, b h j d -> b h i j', Q, K)
        mean_corr= tf.reduce_mean(corr, axis=[1, 2])
        
        # choose top-k
        k = self.max_k if self.max_k is not None else 1
        
        # argsort then take first k indices
        sorted_idx = tf.argsort(mean_corr, direction="DESCENDING")
        topk_idx= sorted_idx[:, :k]
        topk_vals= tf.gather(mean_corr, topk_idx, batch_dims=1)
        weights= tf.nn.softmax(topk_vals ,axis=-1)
        
        # time delay aggregation
        delays_agg= tf.zeros_like(V)
        
        # for each of k delays do per-batch roll and weighted sum
        for i in range(k):
            delays= topk_idx[:, i]
            w= weights[:, i]
            
            # roll V per batch according to delays
            def roll_one(b):
                v_b= V[b]
                shift= -tf.cast(delays[b], tf.int32)    
                return tf.roll(v_b, shift=shift, axis=1)
            rolled= tf.map_fn(roll_one, tf.range(batch_size), dtype=V.dtype)
            w_exp= tf.reshape(w, (batch_size, 1, 1, 1))
            delays_agg += rolled * tf.cast(w_exp, V.dtype)
        
        # Transpose delay
        agg_output= tf.transpose(delays_agg, perm=[0, 2, 1, 3])
        
        # Reshape to (batch_size, seq_len, d_model) 
        agg_output = tf.reshape(agg_output, (batch_size, len_q, self.d_model))
        
        # Final linear projection
        output = tf.tensordot(agg_output, self.wo, axes=[[2],[0]])
        
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