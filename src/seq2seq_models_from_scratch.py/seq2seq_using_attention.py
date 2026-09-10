import torch as pt
import math
import sys

     
sys.path.append(r'C:\THEDO DONT TOUCH\machine learning basics\deep learning')
device =pt.device('cuda' if pt.cuda.is_available() else 'cpu')

class Attention:
    def __init__(self,hidden_dim_enc,input_dim_enc,batch_size_enc,hidden_dim_dec,input_dim_dec,batch_size_dec,vocab_size):
        self.encoder = Attention.Encoder(hidden_dim=hidden_dim_enc,input_dim=input_dim_enc,batch_size=batch_size_enc)
        self.decoder = Attention.Decoder(input_dim=input_dim_dec,hidden_dim=hidden_dim_dec,batch_size=batch_size_dec,encoder_dim=hidden_dim_enc,vocab_size=vocab_size)
        self.network_list = list(self.decoder.net_encoder.parameters())+list(self.decoder.net_prev_hidden.parameters())+list(self.decoder.net_prediction.parameters())+list(self.decoder.net_pre_aplha.parameters())
        self.optimzer_list = [self.encoder.hidden_weight, self.encoder.input_weight, self.encoder.bias, self.decoder.hidden_weight, self.decoder.input_weight, self.decoder.bias]+self.network_list
    class Encoder:
        def __init__(self,hidden_dim,input_dim,batch_size):
            self.hidden_dim = hidden_dim
            self.input_dim = input_dim
            self.batch_size = batch_size
            self.hidden_weight = self.hidden_weight_matrix()
            self.input_weight = self.input_weight_matrix()
            self.bias = self.bias_matrix()
            self.hidden_state = self.hidden_state_matrix()
            self.hidden_state_list =[]
            
            
     
        def hidden_weight_matrix(self):
            initial = pt.empty(size=(self.hidden_dim,self.hidden_dim))
            matrix =  pt.nn.init.xavier_uniform_(tensor=initial).to(device)
            matrix.requires_grad_(True)
            return matrix
        def input_weight_matrix(self):
            initial = pt.empty(size=(self.hidden_dim,self.input_dim))
            matrix = pt.nn.init.xavier_uniform_(tensor=initial).to(device)
            matrix.requires_grad_(True)
            return matrix
        def bias_matrix(self):
            matrix = pt.zeros(size=(self.hidden_dim,)).to(device)
            matrix.requires_grad_(True)
            return matrix
        def hidden_state_matrix(self):
            matrix = pt.zeros(size=(self.batch_size,self.hidden_dim)).to(device)
            return matrix
        
        def padding_encoder(self,input,batch_size):
            fixed_batch =[]
            for i in range(0,len(input),batch_size): 
                chunk= input[i:i+batch_size]
                fixed_batch.append(pt.nn.utils.rnn.pad_sequence(chunk,batch_first=True).to(device))
            return fixed_batch
        
        def encoder_foward(self,input):
                self.hidden_state_list=[]
                self.hidden_state=self.hidden_state_matrix()
  # take one padded 3d  matrix at a time
                for  index in range(input.shape[1]):
                    input_sentences = input[:,index,:]
                    linear_trans = (pt.matmul(input_sentences,self.input_weight.T)) + pt.matmul(self.hidden_state, self.hidden_weight.T) + (self.bias)
                    output = pt.tanh(linear_trans).to(device)
                    self.hidden_state_list.append(output)
                    self.hidden_state = output
                return output,self.hidden_state_list
            
    class Decoder:
        def __init__(self,input_dim,hidden_dim,batch_size,encoder_dim,vocab_size):
            self.hidden_dim = hidden_dim
            self.input_dim = input_dim
            self.batch_size = batch_size
            self.hidden_weight = self.hidden_weight_matrix()
            self.input_weight = self.input_weight_matrix()
            self.bias = self.bias_matrix() 
            self.net_encoder = pt.nn.Linear(in_features=encoder_dim,out_features=256).to(device)
            self.net_prev_hidden = pt.nn.Linear(in_features=hidden_dim,out_features=256).to(device)
            self.net_pre_aplha = pt.nn.Linear(in_features=256,out_features=1).to(device)
            self.net_prediction = pt.nn.Linear(in_features=hidden_dim,out_features=vocab_size).to(device)
            self.prediction_list =[]
            
        def hidden_weight_matrix(self):
            initial = pt.empty(size=(self.hidden_dim,self.hidden_dim))
            matrix =  pt.nn.init.xavier_uniform_(tensor=initial).to(device)
            matrix.requires_grad_(True)
            return matrix
        def input_weight_matrix(self):
            initial = pt.empty(size=(self.hidden_dim,self.input_dim))
            matrix = pt.nn.init.xavier_uniform_(tensor=initial).to(device)
            matrix.requires_grad_(True)
            return matrix
        def bias_matrix(self):
            matrix = pt.zeros(size=(self.hidden_dim,)).to(device)
            matrix.requires_grad_(True)
            return matrix
        
        def alignment_model(self,hidden_state_list,current_hidden_state):
            alpha_list =[]
            prev_hidden=self.net_prev_hidden.forward(current_hidden_state)
            for hidden_state in hidden_state_list:
                hidden_state_position=self.net_encoder.forward(hidden_state)
                combination = hidden_state_position+prev_hidden
                final_net_input = pt.tanh(combination)
                final_net = self.net_pre_aplha.forward(final_net_input)
                alpha_list.append(final_net)
            scalars =pt.stack(alpha_list)
            scalars = pt.softmax(scalars,dim=0)
            context_vector = scalars*pt.stack(hidden_state_list)
            context_vector = pt.sum(context_vector,dim=0)
            return context_vector
                
        
        def padding_decoder(self,input,batch_size):
            fixed_batch =[]
            for i in range(0,len(input),batch_size): 
                chunk= input[i:i+batch_size]
                fixed_batch.append(pt.nn.utils.rnn.pad_sequence(chunk,batch_first=True).to(device))
            return fixed_batch
        
        def decoder_foward(self,input,encoder_hidden_state,hidden_state_list):
                    self.hidden_state = encoder_hidden_state
                    self.prediction_list=[]
                    for i in range(input.shape[1]):
                          input_sentences = input[:,i,:]
                          context_vector = self.alignment_model(hidden_state_list,self.hidden_state)
                          linear_trans = (pt.matmul(input_sentences,self.input_weight.T)) + pt.matmul(self.hidden_state, self.hidden_weight.T) + (self.bias)+context_vector
                          output = pt.tanh(linear_trans).to(device)
                          self.hidden_state = output
                          prediction = self.net_prediction.forward(output)
                          self.prediction_list.append(prediction)
                          
        def label_padding(self,labels,batch_size):
            label_list =[]
            for i in range(0,len(labels),batch_size): 
                chunk= labels[i:i+batch_size]
                label_list.append(pt.nn.utils.rnn.pad_sequence(chunk,batch_first=True).long().to(device))
            return label_list
        def loss(self,labels):
            predictions=pt.stack(self.prediction_list)
            predictions = predictions.permute(1, 2, 0)
            predictions = predictions[:, :, :labels.shape[1]]
            loss_func=pt.nn.CrossEntropyLoss(reduction='mean')
            average_loss=loss_func(predictions,labels)
            return average_loss
            
    def train(self,X_enc,X_dec,label,epoch,learning_rate,hidden_dim_enc,input_dim_enc,batch_size_enc,hidden_dim_dec,input_dim_dec,batch_size_dec,vocab_size):
        loss_track =[]
        gradient_track =[]
        labels= self.decoder.label_padding(label,batch_size_dec)
        encoder_padding = self.encoder.padding_encoder(X_enc,batch_size_enc)
        decoder_padding = self.decoder.padding_decoder(X_dec,batch_size_dec)
        adam=pt.optim.Adam(self.optimzer_list,lr=learning_rate)
        count =0
        for epochs in range(epoch):
            count +=1
            print(f'epoch :{count}')
            counts = 0
            for batch_enc,batch_dec,batch_y in zip(encoder_padding,decoder_padding,labels):
                counts+=1
                print(f'batches: {counts}')
                adam.zero_grad()
                final_hidden,hidden_states_positions = self.encoder.encoder_foward(batch_enc)
                self.decoder.decoder_foward(batch_dec,final_hidden,hidden_states_positions)
                loss = self.decoder.loss(batch_y)
                loss_track.append(loss.item())
                loss.backward()
                gradient_track.append(self.decoder.net_prediction.weight.grad.norm().item())
                adam.step()
        return loss_track,gradient_track
                
      
english_tensors = pt.load('data/english_embedded.pt')
french_tensors = pt.load('data/french_embedded.pt')
label_tensors = pt.load('data/french_labels.pt')


model = Attention(
    hidden_dim_enc=256, input_dim_enc=256, batch_size_enc=64,
    hidden_dim_dec=256, input_dim_dec=256, batch_size_dec=64,
    vocab_size=17005
)


english_tensors = english_tensors[:28992]
french_tensors = french_tensors[:28992]
label_tensors = label_tensors[:28992]

pt.save({
    'encoder_hidden_weight': model.encoder.hidden_weight,
    'encoder_input_weight': model.encoder.input_weight,
    'encoder_bias': model.encoder.bias,
    'decoder_hidden_weight': model.decoder.hidden_weight,
    'decoder_input_weight': model.decoder.input_weight,
    'decoder_bias': model.decoder.bias,
    'net_encoder': model.decoder.net_encoder.state_dict(),
    'net_prev_hidden': model.decoder.net_prev_hidden.state_dict(),
    'net_pre_alpha': model.decoder.net_pre_aplha.state_dict(),
    'net_prediction': model.decoder.net_prediction.state_dict(),
}, 'attention_model.pt')
