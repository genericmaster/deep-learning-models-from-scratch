import numpy as np
import sys
sys.path.append(r'C:\THEDO DONT TOUCH\machine learning basics\deep learning')
from decoder import Decoder
from encoder import Encoder

class seq2seq:
 def __init__(self,hidden_dim,input_dim,loss):
  self.encoder = Encoder(hidden_dim=hidden_dim,input_dim=input_dim,loss=loss)
  self.decoder = Decoder(input_dim=input_dim,hidden_dim=hidden_dim,loss =loss)
  
 def train(self,labels_dec,input_enc,input_dec,epoch=50,learning_rate=0.01,optimizer ='adam'):
        count =0
        for epochs in range(epoch):
            for enc_sentence, dec_sentence, label in zip(input_enc, input_dec, labels_dec):
                 count+=1
                 print(count)
                 label_list =[]
                 self.encoder.encoder.forward(enc_sentence)
                 encoder_hidden = self.encoder.encoder.hidden_state
                 self.encoder.encoder.hidden_state = self.encoder.encoder.init_hidden_state
             
                 for truth_label in label:
                        
                     
                        vector=np.zeros(shape=(17005,), dtype=np.float32)
                        vector[truth_label] = 1
                        label_list.append(vector)
               
                 self.decoder.train(encoder_hidden_state=encoder_hidden, X=dec_sentence, label=label_list, learning_rate=learning_rate, optimizer=optimizer)

                 
english_embedded = list(np.load('data/english_embedded.npy', allow_pickle=True))
french_embedded = list(np.load('data/french_embedded.npy', allow_pickle=True))
french_labels = list(np.load('data/french_label_indices.npy', allow_pickle=True))


model = seq2seq(hidden_dim=128, input_dim=256, loss='categorical_cross_entropy')
model.decoder.decoder.net.result_layer(neurons=17005, features=128, activation='softmax')


model.train(labels_dec=french_labels[:1000], input_enc=english_embedded[:1000], input_dec=french_embedded[:1000], epoch=10,learning_rate=0.0001)
print(model.decoder.decoder.net.loss_track)