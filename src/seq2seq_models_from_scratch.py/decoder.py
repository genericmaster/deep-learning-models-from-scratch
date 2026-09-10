import numpy as np
import math
import sys 
sys.path.append(r'C:\THEDO DONT TOUCH\machine learning basics\deep learning')
from neural_network_from_scratch.full_neural_net import Neural_net

class Decoder:
    def __init__(self,input_dim,hidden_dim,loss):
        self.decoder = self.Decoder_layer(input_dim=input_dim,hidden_dim=hidden_dim,loss =loss)

    class Decoder_layer:
        def __init__(self,input_dim,hidden_dim,loss):
            self.input_size = input_dim
            self.hidden_size = hidden_dim
            self.bias_matrix = self.bias_matrix_init()
            self.input_weight_matrix = self.input_weight_init()
            self.hidden_weight_matrix = self.hidden_weight_init()
            self.hidden_state = None
            self.net = Neural_net(loss=loss)
            self.loss_track =[]
            self.input_list =[]
            self.hidden_state_list =[]
            self.hidden_state_init = None
            
        def input_weight_init(self):
            input_weight = np.random.uniform(low=-math.sqrt(6/(self.input_size+self.hidden_size)),high=math.sqrt(6/(self.input_size+self.hidden_size)),size=(self.hidden_size,self.input_size))
            return input_weight
        
        def hidden_weight_init(self):
            hidden_weight = np.random.uniform(low=-math.sqrt(6/(2*self.hidden_size)),high=math.sqrt(6/(2*self.hidden_size)),size=(self.hidden_size,self.hidden_size))
            return hidden_weight
        def bias_matrix_init(self):
            bias = np.zeros(shape=(self.hidden_size,))
            return bias
        
        def forward(self,input,encoder_hidden_state):
            self.prediction_list = []
            self.input_list =[]
            self.hidden_state_list =[]
            self.hidden_state = encoder_hidden_state
            self.hidden_state_init = encoder_hidden_state
            for index in range(input.shape[0]):
                self.input_list.append(input[index])
                linear_trans = (self.input_weight_matrix@input[index]) + (self.hidden_weight_matrix@self.hidden_state)+ (self.bias_matrix)
                output = (np.exp(linear_trans)-np.exp(-linear_trans))/(np.exp(linear_trans)+np.exp(-linear_trans))
                self.hidden_state= output
                self.hidden_state_list.append(output)
                prediction =self.net.forward(output)
                self.prediction_list.append(prediction)
            return prediction
                
        def decoder_loss(self,labels):
            sum =0
            if np.array_equal(np.unique(labels), [0,1]):
                for prediction,truth_label in zip(self.prediction_list,labels):
                        categorical_cross_entropy = -(np.sum(np.sum(truth_label*np.log(np.clip(prediction, 1e-7, 1)))))
                        sum+= float(categorical_cross_entropy)
            else:
                raise ValueError("categorical_cross_entropy only requires inputs of 0 or 1")
            average_loss = sum/len(self.prediction_list)
            self.loss_track.append(average_loss)
            return average_loss
        def backpropagation(self,incoming_gradient,hidden_state,prev_hidden_state,input):
            stateless_component = incoming_gradient*(1- (hidden_state)**2)
            hidden_state_component = np.outer(stateless_component,prev_hidden_state)
            input_component = np.outer(stateless_component,input)
            bias_component = stateless_component 
            outgoing_gradient = self.hidden_weight_matrix@stateless_component
            return outgoing_gradient,input_component,hidden_state_component,bias_component
        
        def BPTT(self,incoming_grad):
            incoming_grad = incoming_grad.squeeze()
            self.W_hh_accum =0
            self.W_xh_accum =0
            self.b_accum =0
            for t in range(len(self.hidden_state_list)-1,-1,-1):
                x_t = self.input_list[t]
                h_t = self.hidden_state_list[t]
                if t==0:
                    prev_hidden_state = self.hidden_state_init
                else:
                     prev_hidden_state=self.hidden_state_list[t-1]
                outgoing_grad, W_xh_contrib, W_hh_contrib, b_contrib=self.backpropagation(incoming_grad, h_t, prev_hidden_state, x_t)
                self.W_xh_accum+=W_xh_contrib
                self.W_hh_accum += W_hh_contrib
                self.b_accum +=b_contrib
                incoming_grad =outgoing_grad
                print('timestep grad norm:', np.linalg.norm(incoming_grad))
            return incoming_grad
        
        def decoder_optimzers(self,learning_rate,optimizer):
             if not hasattr(self, 'velocity_weight_hidden_momentum'):
                self.velocity_weight_hidden_momentum = np.zeros_like(self.hidden_weight_matrix)
                self.velocity_weight_input_momentum = np.zeros_like(self.input_weight_matrix)
                self.velocity_weight_bias_momentum = np.zeros_like(self.bias_matrix)
                self.velocity_weight_hidden_rmsprop = np.zeros_like(self.hidden_weight_matrix)
                self.velocity_weight_input_rmsprop = np.zeros_like(self.input_weight_matrix)
                self.velocity_weight_bias_rmsprop = np.zeros_like(self.bias_matrix)
                self.m_weight_hidden_adam = np.zeros_like(self.hidden_weight_matrix)
                self.m_weight_input_adam = np.zeros_like(self.input_weight_matrix)
                self.m_weight_bias_adam = np.zeros_like(self.bias_matrix)
                self.m_weight_hidden_momentum_adam = np.zeros_like(self.hidden_weight_matrix)
                self.m_weight_input_momentum_adam = np.zeros_like(self.input_weight_matrix)
                self.m_weight_bias_momentum_adam = np.zeros_like(self.bias_matrix)
                self.t =0
            
             if optimizer =='sgd':
                weight_h = self.hidden_weight_matrix- learning_rate*self.W_hh_accum
                self.hidden_weight_matrix =weight_h
                weight_xh = self.input_weight_matrix - learning_rate * self.W_xh_accum
                self.input_weight_matrix = weight_xh
                weight_bias = self.bias_matrix - learning_rate*self.b_accum
                self.bias_matrix = weight_bias
             if optimizer == 'momentum':
                velo_t_momentum_h =0.9*self.velocity_weight_hidden_momentum+(1-0.9)*self.W_hh_accum
                weight_h =  self.hidden_weight_matrix- learning_rate*velo_t_momentum_h
                self.hidden_weight_matrix = weight_h
                self.velocity_weight_hidden_momentum = velo_t_momentum_h
                velo_t_momentum_xh = 0.9*self.velocity_weight_input_momentum+(1-0.9)*self.W_xh_accum
                weight_xh = self.input_weight_matrix - learning_rate * velo_t_momentum_xh
                self.input_weight_matrix =weight_xh
                self.velocity_weight_input_momentum = velo_t_momentum_xh
                velo_t_bias = 0.9*self.velocity_weight_bias_momentum +(1-0.9)*self.b_accum
                weight_bias = self.bias_matrix - learning_rate*velo_t_bias
                self.bias_matrix = weight_bias
                self.velocity_weight_bias_momentum = velo_t_bias
                
             if optimizer =='rmsprop':
                velo_t_rmsprop_h =0.9*self.velocity_weight_hidden_rmsprop+(1-0.9)*(self.W_hh_accum**2)
                weight_h =  self.hidden_weight_matrix- learning_rate*(self.W_hh_accum/np.sqrt(velo_t_rmsprop_h+1e-08))
                self.hidden_weight_matrix = weight_h
                self.velocity_weight_hidden_rmsprop = velo_t_rmsprop_h
                velo_t_rmsprop_xh = 0.9*self.velocity_weight_input_rmsprop+(1-0.9)*(self.W_xh_accum**2)
                weight_xh = self.input_weight_matrix - learning_rate * (self.W_xh_accum/np.sqrt(velo_t_rmsprop_xh+1e-08))
                self.input_weight_matrix =weight_xh
                self.velocity_weight_input_rmsprop= velo_t_rmsprop_xh
                velo_t_bias_rmsprop = 0.9*self.velocity_weight_bias_rmsprop +(1-0.9)*(self.b_accum**2)
                weight_bias = self.bias_matrix - learning_rate*(self.b_accum/np.sqrt(velo_t_bias_rmsprop+1e-08))
                self.bias_matrix = weight_bias
                self.velocity_weight_bias_rmsprop = velo_t_bias_rmsprop
             self.t+=1
             if optimizer =='adam':
                #momentm logic
                velo_t_adam_h =0.9*self.m_weight_hidden_momentum_adam+(1-0.9)*self.W_hh_accum
                self.m_weight_hidden_momentum_adam = velo_t_adam_h
                velo_t_adam_xh = 0.9*self.m_weight_input_momentum_adam+(1-0.9)*self.W_xh_accum
                self.m_weight_input_momentum_adam = velo_t_adam_xh
                velo_t_adam_bias = 0.9*self.m_weight_bias_momentum_adam +(1-0.9)*self.b_accum
                self.m_weight_bias_momentum_adam = velo_t_adam_bias
                
                #velocity logic
                velo_t_v_adam_h =0.99*self.m_weight_hidden_adam+(1-0.99)*(self.W_hh_accum**2)
                self.m_weight_hidden_adam = velo_t_v_adam_h
                velo_t_v_adam_xh = 0.99*self.m_weight_input_adam+(1-0.99)*(self.W_xh_accum**2)
                self.m_weight_input_adam =  velo_t_v_adam_xh
                velo_t_v_bias_adam = 0.99*self.m_weight_bias_adam +(1-0.99)*(self.b_accum**2)
                self.m_weight_bias_adam = velo_t_v_bias_adam
                
                #bias correction
                m_weight_hidden_momentum_adam_corr = self.m_weight_hidden_momentum_adam/((1-(0.9**self.t)))
                m_weight_input_momentum_adam_corr =  self.m_weight_input_momentum_adam/((1-(0.9**self.t)))
                m_weight_bias_momentum_adam_corr = self.m_weight_bias_momentum_adam/((1-(0.9**self.t)))
                m_weight_hidden_adam_corr= self.m_weight_hidden_adam/((1-(0.99**self.t)))
                m_weight_input_adam_corr = self.m_weight_input_adam/((1-(0.99**self.t)))
                m_weight_bias_adam_corr = self.m_weight_bias_adam/((1-(0.99**self.t)))
                
                #weight update
                self.hidden_weight_matrix= self.hidden_weight_matrix- learning_rate *(m_weight_hidden_momentum_adam_corr/np.sqrt( m_weight_hidden_adam_corr+1e-08))
                self.input_weight_matrix = self.input_weight_matrix - learning_rate *(m_weight_input_momentum_adam_corr/np.sqrt( m_weight_input_adam_corr+1e-08))
                self.bias_matrix = self.bias_matrix - learning_rate *(m_weight_bias_momentum_adam_corr/np.sqrt( m_weight_bias_adam_corr+1e-08))
    def train(self,encoder_hidden_state,X,label,learning_rate =0.001,optimizer='sgd'):
                  self.decoder.net.prediction_list =[]
                  self.decoder.forward(X,encoder_hidden_state)
                  incoming_grad =0
                  sum_pred =0
                  sum_truth =0
                  for truth,prediction in zip(label,self.decoder.net.prediction_list):
                      sum_pred+=prediction
                      sum_truth += truth
                  average_pred = sum_pred/len(self.decoder.net.prediction_list) 
                  average_truth = sum_truth/len(label)
                  incoming_grad =self.decoder.net.backward(average_truth,average_pred,loss_type='normal')
                    
                  self.decoder.BPTT(incoming_grad)
                  self.decoder.net.optimizers(learning_rate,optimizer)
                  self.decoder.decoder_optimzers(learning_rate,optimizer)
            
               
                
            