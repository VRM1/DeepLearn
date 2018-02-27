from keras.layers import Dense, Input, Lambda, Embedding, Flatten
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np
from tqdm import tqdm
from keras.backend import int_shape
from keras import metrics
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from collections import defaultdict
from keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

# # this is the class for creating the DSAE with embeding layer
# class LinkModel:
#     
#     def __init__(self,inter_dim, z_dim, bath_size,out_dim):
#         
#         self.inpt_dim = 200
#         self.vocab_size = 5004
#         self.inter_dim = inter_dim
#         self.z_dim = z_dim
#         self.bath_size = bath_size
#         self.epochs = 50
#         self.out_dim = out_dim
#         
#     def _dsae_loss(self,z_mean,z_vari):
#         def loss(x, x_decoded_mean):
#             """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
#             # E[log P(X^{a}|z)]
#             recon_a = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=1)
#             # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
#             kl_a = 0.5 * K.sum(K.exp(z_vari) + K.square(z_mean) - 1. - z_vari, axis=1)
#             
#             fin_loss = recon_a + kl_a
#             return fin_loss
#         return loss
#     def VAE(self):
#     
#         # the number of nodes for the intermediate dimension
#         epsilon_std = 1.0
#         # mean and variance for isotropic normal distribution
#         
#         # create a place holder for input layer
#         x = Input(shape=(self.inpt_dim,))
#         x_embed = Embedding(self.vocab_size, 64, input_length=self.inpt_dim)(x)
#         # flatten the embeding layer see: 
#         flat_x = Flatten()(x_embed)
#         g = Dense(self.inter_dim,activation='relu')(flat_x)
#         z_mean = Dense(self.z_dim)(g)
#         z_vari = Dense(self.z_dim)(g)
#         
#         # create the latent layer
#         z = Lambda(self._SampleZ,output_shape=(self.z_dim,), arguments={'epsilon_std':epsilon_std, 'batch_size':self.bath_size,
#                                                               'latent_dim':self.z_dim})([z_mean,z_vari])
#         # decoder, intermediate layer
#         decoded_g = Dense(self.inter_dim,activation='relu')
#         
#         decoded_mean = Dense(self.out_dim, activation='sigmoid')
#         g_decoded = decoded_g(z)
#         x_decoded_mean = decoded_mean(g_decoded)
#         # create the complete vae model
#         vae = Model(x,x_decoded_mean)
#         # create just the encoder
#         encoder = Model(x, z_mean)
#     
#         return (vae,x,z_mean,z_vari,z,x_decoded_mean,encoder)
# 
# 
#     def _SampleZ(self, args,epsilon_std,latent_dim,batch_size):
#         
#         z_mean, z_var = args
#         # sample Z from isotropic normal
#         epslon = K.random_normal(shape=(1, latent_dim), mean=0., stddev=epsilon_std)
#         return z_mean + z_var * epslon
# 
# 
#     # method of the classifier neural network
#     def _ClassifierNN(self,z_a,z_b):
#         
#         # concatenate the two latent layers from VAE_a and VAE_b
#         conc = concatenate([z_a, z_b])
#         # get the final dimension of the concatenated object
#         g_za_zb = Dense(self.inter_dim,activation='relu')(conc)
#         # we have four labels 0,1 are for substitutes and 2,3 are for compliments 
#         y_label = Dense(2,activation='softmax')(g_za_zb)
# 
#         return y_label
#     
#     # method that get the DSAE model
#     def getModels(self):
#         
#         # create the first VAE model (i.e., vae_a)
#         vae_a,x_a,z_mean_a,z_vari_a,z_a,x_decoded_mean,encoder = self.VAE()
#         #     create the second VAE model (i.e., vae_b)
#         vae_b,x_b,z_mean_b,z_vari_b,z_b,x_decoded_mean,encoder = self.VAE()
#         # create the classifier neural network
#         classifier_nn = self._ClassifierNN(z_a,z_b)
#         # create the complete model
#         DSAE = Model(inputs=[vae_a.inputs[0],vae_b.inputs[0]],\
#                      outputs=[vae_a.outputs[0],vae_b.outputs[0],classifier_nn])
#         
#         #     Compile the autoencoder computation graph
#         LinkPredictor = Model(inputs=[DSAE.inputs[0],DSAE.inputs[1]],outputs=[DSAE.outputs[2]])
#         # model for evaluation
#         DSAE.compile(optimizer="adam", loss=[self._dsae_loss(z_mean_a,z_vari_a), \
#                                              self._dsae_loss(z_mean_b,z_vari_b),\
#                                              'binary_crossentropy'])
#         LinkPredictor.compile(optimizer="adam", loss='binary_crossentropy',metrics=['accuracy'])
#         plot_model(DSAE,to_file='DSAE.png',show_shapes=True)
#         plot_model(LinkPredictor,to_file='LinkPredictor.png',show_shapes=True)
#         return (DSAE,LinkPredictor)


# this is the class for creating the DSAE with embeding layer
class DSAEB:
    
    def __init__(self,name,inter_dim, z_dim, inpt_dim,bath_size,typ):
        
        self.inpt_dim = inpt_dim
        self.inter_dim = inter_dim
        self.z_dim = z_dim
        self.bath_size = bath_size
        self.epochs = 50
        self.name = name
        self.typ = typ
    def _dsae_loss(self,z_mean,z_vari):
        def loss(x, x_decoded_mean):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            # E[log P(X^{a}|z)]
            recon_a = self.inpt_dim * metrics.mean_squared_error(x, x_decoded_mean)
            # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl_a = - 0.5 * K.sum(1 + z_vari - K.square(z_mean) - K.exp(z_vari), axis=-1)            
            fin_loss = recon_a + kl_a
            return fin_loss
        return loss
    
    def VAE(self):
    
        # the number of nodes for the intermediate dimension
        epsilon_std = 1.0
        # mean and variance for isotropic normal distribution
        
        # create a place holder for input layer
        x = Input(shape=(self.inpt_dim,))
        g = Dense(self.inter_dim,activation='relu')(x)
        z_mean = Dense(self.z_dim)(g)
        z_vari = Dense(self.z_dim)(g)
        
        # create the latent layer
        z = Lambda(self._SampleZ,output_shape=(self.z_dim,), arguments={'epsilon_std':epsilon_std, 'batch_size':self.bath_size,
                                                              'latent_dim':self.z_dim})([z_mean,z_vari])
        # decoder, intermediate layer
        decoded_g = Dense(self.inter_dim,activation='relu')
        
        decoded_mean = Dense(self.inpt_dim)
        g_decoded = decoded_g(z)
        x_decoded_mean = decoded_mean(g_decoded)
        # create the complete vae model
        vae = Model(x,x_decoded_mean)
        # create just the encoder
        encoder = Model(x, z_mean)
    
        return (vae,x,z_mean,z_vari,z,x_decoded_mean,encoder)


    def _SampleZ(self, args,epsilon_std,latent_dim,batch_size):
        
        z_mean, z_var = args
        # sample Z from isotropic normal
        epslon = K.random_normal(shape=(1, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + z_var * epslon


    # method of the classifier neural network
    def _ClassifierNN(self,z_a,z_b):
        
        if self.typ == 'binary':
            lbl = 2
        else:
            lbl = 4
        # concatenate the two latent layers from VAE_a and VAE_b
        conc = concatenate([z_a, z_b])
        # get the final dimension of the concatenated object
        g_za_zb = Dense(self.inter_dim,activation='relu')(conc)
        # we have four labels 0,1 are for substitutes and 2,3 are for compliments 
        y_label = Dense(lbl,activation='softmax')(g_za_zb)

        return y_label
    
    # method that get the DSAE model
    def getModels(self):
        
        # create the first VAE model (i.e., vae_a)
        vae_a,x_a,z_mean_a,z_vari_a,z_a,x_decoded_mean,encoder = self.VAE()
        #     create the second VAE model (i.e., vae_b)
        vae_b,x_b,z_mean_b,z_vari_b,z_b,x_decoded_mean,encoder = self.VAE()
        # create the classifier neural network
        classifier_nn = self._ClassifierNN(z_a,z_b)
        # create the complete model
        DSAE = Model(inputs=[vae_a.inputs[0],vae_b.inputs[0]],\
                     outputs=[vae_a.outputs[0],vae_b.outputs[0],classifier_nn])
        
        #     Compile the autoencoder computation graph
        LinkPredictor = Model(inputs=[DSAE.inputs[0],DSAE.inputs[1]],outputs=[DSAE.outputs[2]])
        # model for evaluation
        DSAE.compile(optimizer="adam", loss=[self._dsae_loss(z_mean_a,z_vari_a), \
                                             self._dsae_loss(z_mean_b,z_vari_b),\
                                             'binary_crossentropy'], loss_weights=[0.3,0.3,0.9])
        
#         DSAE.compile(optimizer="adam", loss=[self._dsae_loss(z_mean_a,z_vari_a), \
#                                              self._dsae_loss(z_mean_b,z_vari_b),\
#                                              'binary_crossentropy'])
        cp = [ModelCheckpoint(filepath='output/DSAE_'+self.name+'.hdf5', verbose=1, monitor='val_loss', mode='min',\
                              save_best_only=True)]
        LinkPredictor.compile(optimizer="adam", loss='binary_crossentropy',metrics=['accuracy'])
#         plot_model(DSAE,to_file='DSAE.png',show_shapes=True)
#         plot_model(LinkPredictor,to_file='LinkPredictor.png',show_shapes=True)
        return (DSAE,LinkPredictor,cp)
    
    def TestVAE(self):
        
        # create the first VAE model (i.e., vae_a)
        vae_a,x_a,z_mean_a,z_vari_a,z_a,x_decoded_mean,encoder = self.VAE()
        vae_a.compile(optimizer='adam', loss=self._dsae_loss(z_mean_a,z_vari_a))
        plot_model(vae_a,to_file='VAE.png',show_shapes=True)
        return vae_a