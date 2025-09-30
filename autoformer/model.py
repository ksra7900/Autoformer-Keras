from keras import layers
from blocks import encoder, decoder

class Autoformer(layers.Layer):
    def __init__(self,
                 d_out,
                 d_model,
                 n_heads,
                 conv_filter,
                 num_decoder= 1,
                 num_encoder= 1,
                 activation='relu',
                 kernel_size= 25,
                 padding= 'same',
                 c=1,
                 **kwargs):
        super(Autoformer, self).__init__(**kwargs)
        self.kernel_size= kernel_size
        self.padding= padding
        self.c= c
        self.d_model= d_model
        self.d_out= d_out
        self.n_heads= n_heads
        self.activation= activation
        self.conv_filter= conv_filter
        self.num_decoder= num_decoder
        self.num_encoder= num_encoder
        # initial encoder&decoder
        self.encoder_layer= [
                            encoder(
                                        d_model = self.d_model,
                                        n_heads = self.n_heads,
                                        conv_filter = self.conv_filter
                                    )for _ in range(self.num_encoder)
                            ]
        
        self.decoder_layer= [
                                decoder(
                                            d_out = self.d_out,
                                            d_model = self.d_model,
                                            n_heads = self.n_heads,
                                            conv_filter = self.conv_filter
                                        )for _ in range(self.num_decoder)
                                ]
        
        self.output_dence= layers.Dense(1, activation='linear')
        
    def build(self, input_shape):
        super().build(input_shape)
        # build encoder layer
        for layer in self.encoder_layer:
            layer.build(input_shape)
        
        # build decoder layer
        for layer in self.decoder_layer:
            layer.build(input_shape)
        
    def call(self, inputs):
        # initial encoder
        x= inputs
        for encoder_layer in self.encoder_layer:
            x= encoder_layer(x)
            
        # initial decoder
        decoder_output= x
        for decoder_layer in self.decoder_layer:
            decoder_output= decoder_layer((decoder_output, x))
            
        # output layer
        output= self.output_dence(decoder_output)
        
        return output
            
    def get_config(self):
        config= super().get_config()
        config.update({
            'd_out' : self.d_out,
            'd_model' : self.d_model,
            'n_heads' : self.n_heads,
            'conv_filter' : self.conv_filter,
            'num_decoder' : self.num_decoder,
            'num_encoder' : self.num_encoder,
            'activation' : self.activation,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'c' : self.c
            })
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)