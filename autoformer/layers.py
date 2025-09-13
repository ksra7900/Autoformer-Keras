import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, AveragePooling1D, Input, Dense

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
        self.wq= self.add_weight(shape=(self.d_model, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wq')
        self.wk= self.add_weight(shape=(self.d_model, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wk')
        self.wv= self.add_weight(shape=(self.d_model, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wv')
        # Outout forecast matrices
        self.wo= self.add_weight(shape=(self.d_model, self.d_model),
                                 initializer='glorot_uniform',
                                 trainable= True,
                                 name= 'wo')
        
        super(AutoCorrelation, self).build(input_shape)
        
    def roll_function(self, v, tau):
        batch_size, n_heads, seq_len, d_head= tf.shape(v)[0], tf.shape(v)[1], tf.shape(v)[2], tf.shape(v)[3]
        
        v_flat= tf.reshape(v, (batch_size * n_heads, seq_len, d_head))
        tau_flat= tf.reshape(tau, (batch_size * n_heads, seq_len))
        
        def roll_single_sequence(args):
            v_vec, tau_vec= args
            
            rolled_sequence= []
            for t in range(seq_len):
                shift= tau_vec[t]
                rolled_v= tf.roll(v_vec, shift=shift, axis=0)
                rolled_sequence.append(rolled_v[t])
                
            return tf.stack(rolled_sequence, axis=0)
        
        output_flat= tf.vectorized_map(roll_single_sequence, (v_flat, tau_flat))
        output= tf.reshape(output_flat, (batch_size, n_heads, seq_len, d_head))
        
        return output
        
    def call(self, queries, keys, values):
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
        Q = tf.transpose(Q, perm=[0, 2, 3, 1])
        K = tf.transpose(K, perm=[0, 2, 3, 1])
        V = tf.transpose(V, perm=[0, 2, 3, 1])
        
        # compute FFT 
        Q_fft = tf.signal.rfft(tf.cast(Q, tf.complex64))
        K_fft = tf.signal.rfft(tf.cast(K, tf.complex64))
        
        # Compute cross-correlation via inverse FFT of element-wise product
        S = Q_fft * tf.math.conj(K_fft)
        corr = tf.signal.irfft(S)
        corr = tf.math.real(corr)
        
        # find top K delay
        k = tf.cast(tf.math.floor(self.c * tf.math.log(tf.cast(len_q, tf.float32))), tf.int32)
        
        # Get top-k values and indices along the last dimension (time lags)
        top_k_vals, top_k_indices = tf.math.top_k(corr, k, sorted= True)

        # Apply softmax to top_k_vals to get weights
        weights = tf.nn.softmax(top_k_vals, axis= -1)
        
        # Delay Aggregation
        output= tf.zeros_like(V)
        
        for i in range(k):
            tau= top_k_indices[..., i]
            w= weights[..., i]
            
            rolled_v= self.roll_function(V, tau)
            
            w= tf.expand_dims(w, axis=-1)
            output += w * rolled_v
            
        output= tf.transpose(output, perm=[0, 2, 1, 3])
        output= tf.reshape(output, (batch_size, len_q, self.d_model))
        
        # create output
        output= tf.matmul(output, self.wo)
        return output

if __name__ == "__main__":
    input_layer = Input(shape=(100, 5))  # (seq_len=100, features=5)
    seasonal, trend = SeriesDecomposition(kernel_size=25)(input_layer)
    
    x = Dense(64)(seasonal)
    output_layer = Dense(1)(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()