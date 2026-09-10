import numpy as np
import math
import sys
sys.path.append(r'C:\THEDO DONT TOUCH\machine learning basics\deep learning')
from neural_network_from_scratch.full_neural_net import Neural_net

class Recurrent_net:
    def __init__(self,input_dim,hidden_dim,loss):
            self.rnn = self.Recurrent_Net_layer(input_dim,hidden_dim)
            self.input= input_dim
            self.hidden = hidden_dim
            self.loss = loss
            self.net = Neural_net(loss=loss)
    class Recurrent_Net_layer:
        def __init__(self,input_dim,hidden_state_dim):
            self.size =input_dim
            self.hidden_state_size = hidden_state_dim
            self.weight_matrix = self.weight_matrix_initialization()
            self.bias_matrix = self.bias_matrix_initalization()
            self.hidden_state_weight_matrix = self.hidden_state_weight_initialization()
            self.hidden_state= self.hidden_state_init()
            self.init_hidden_state =self.hidden_state_init()
            self.linear_trans_weight = None
            self.input_list = []
            self.hidden_state_list = []
            
        def weight_matrix_initialization(self):
            weight = np.random.uniform(low=-math.sqrt(6/(self.size+self.hidden_state_size)),high=math.sqrt(6/(self.size+self.hidden_state_size)),size=(self.hidden_state_size,self.size))
            return weight

        def bias_matrix_initalization(self):
            bias = np.zeros(shape=(self.hidden_state_size,))
            return bias
        
        def hidden_state_weight_initialization(self):
            hidden_state_matrix = np.random.uniform(low=-math.sqrt(6/(2*self.hidden_state_size)),high=math.sqrt(6/(2*self.hidden_state_size)),size=(self.hidden_state_size,self.hidden_state_size))
            print(hidden_state_matrix.shape)
            return hidden_state_matrix
        
        def hidden_state_init(self):
            hidden_state = np.zeros(shape=(self.hidden_state_size,))
            return hidden_state
            
        def forward(self,input):
            for  recure in range(input.shape[0]):
                self.input_list.append(input[recure])
                linear_trans = (self.hidden_state@self.hidden_state_weight_matrix)+(self.weight_matrix@input[recure]+ (self.bias_matrix))
                activation = (np.exp(linear_trans)-np.exp(-linear_trans))/(np.exp(linear_trans)+np.exp(-linear_trans))
                self.hidden_state= activation
                self.hidden_state_list.append(activation)
            self.output = activation
            return activation
           
        def backpropagation(self,incoming_grad,input,hidden_state,prev_hidden_state):
            activation_deriv = (1- (hidden_state)**2)
            stateless_component =  incoming_grad*activation_deriv
            hidden_state_component = np.outer(stateless_component,prev_hidden_state)
            input_component = np.outer(stateless_component,input)
            outgoing_gradient = self.hidden_state_weight_matrix@ stateless_component
            bias_component = stateless_component
            return outgoing_gradient,input_component,hidden_state_component,bias_component
        
        def BPTT(self,incoming_grad):
            self.W_xh_grad = 0
            self.W_hh_grad = 0
            self.b_grad = 0
            for t in range(len(self.input_list)-1,-1,-1):
                x_t = self.input_list[t]
                h_t = self.hidden_state_list[t]
                if t==0:
                    prev_hidden_state = self.init_hidden_state
                else:
                    prev_hidden_state=self.hidden_state_list[t-1]
                outgoing_grad, W_xh_contrib, W_hh_contrib, b_contrib=self.backpropagation(incoming_grad,x_t,h_t,prev_hidden_state)
                self.W_xh_grad+=W_xh_contrib
                self.W_hh_grad += W_hh_contrib
                self.b_grad +=b_contrib
                incoming_grad =outgoing_grad
            return incoming_grad
        def  rnn_optimizer(self,learning_rate,optimizer):
            if not hasattr(self, 'velocity_weight_hidden_momentum'):
                self.velocity_weight_hidden_momentum = np.zeros_like(self.hidden_state_weight_matrix)
                self.velocity_weight_input_momentum = np.zeros_like(self.weight_matrix)
                self.velocity_weight_bias_momentum = np.zeros_like(self.bias_matrix)
                self.velocity_weight_hidden_rmsprop = np.zeros_like(self.hidden_state_weight_matrix)
                self.velocity_weight_input_rmsprop = np.zeros_like(self.weight_matrix)
                self.velocity_weight_bias_rmsprop = np.zeros_like(self.bias_matrix)
                self.m_weight_hidden_adam = np.zeros_like(self.hidden_state_weight_matrix)
                self.m_weight_input_adam = np.zeros_like(self.weight_matrix)
                self.m_weight_bias_adam = np.zeros_like(self.bias_matrix)
                self.m_weight_hidden_momentum_adam = np.zeros_like(self.hidden_state_weight_matrix)
                self.m_weight_input_momentum_adam = np.zeros_like(self.weight_matrix)
                self.m_weight_bias_momentum_adam = np.zeros_like(self.bias_matrix)
                self.t =0
                
                
            
            if optimizer =='sgd':
                weight_h = self.hidden_state_weight_matrix- learning_rate*self.W_hh_grad
                self.hidden_state_weight_matrix =weight_h
                weight_xh = self.weight_matrix - learning_rate * self.W_xh_grad
                self.weight_matrix = weight_xh
                weight_bias = self.bias_matrix - learning_rate*self.b_grad
                self.bias_matrix = weight_bias
            if optimizer == 'momentum':
                velo_t_momentum_h =0.9*self.velocity_weight_hidden_momentum+(1-0.9)*self.W_hh_grad
                weight_h =  self.hidden_state_weight_matrix- learning_rate*velo_t_momentum_h
                self.hidden_state_weight_matrix = weight_h
                self.velocity_weight_hidden_momentum = velo_t_momentum_h
                velo_t_momentum_xh = 0.9*self.velocity_weight_input_momentum+(1-0.9)*self.W_xh_grad
                weight_xh = self.weight_matrix - learning_rate * velo_t_momentum_xh
                self.weight_matrix =weight_xh
                self.velocity_weight_input_momentum = velo_t_momentum_xh
                velo_t_bias = 0.9*self.velocity_weight_bias_momentum +(1-0.9)*self.b_grad
                weight_bias = self.bias_matrix - learning_rate*velo_t_bias
                self.bias_matrix = weight_bias
                self.velocity_weight_bias_momentum = velo_t_bias
                
            if optimizer =='rmsprop':
                velo_t_rmsprop_h =0.9*self.velocity_weight_hidden_rmsprop+(1-0.9)*(self.W_hh_grad**2)
                weight_h =  self.hidden_state_weight_matrix- learning_rate*(self.W_hh_grad/np.sqrt(velo_t_rmsprop_h+1e-08))
                self.hidden_state_weight_matrix = weight_h
                self.velocity_weight_hidden_rmsprop = velo_t_rmsprop_h
                velo_t_rmsprop_xh = 0.9*self.velocity_weight_input_rmsprop+(1-0.9)*(self.W_xh_grad**2)
                weight_xh = self.weight_matrix - learning_rate * (self.W_xh_grad/np.sqrt(velo_t_rmsprop_xh+1e-08))
                self.weight_matrix =weight_xh
                self.velocity_weight_input_rmsprop= velo_t_rmsprop_xh
                velo_t_bias_rmsprop = 0.9*self.velocity_weight_bias_rmsprop +(1-0.9)*(self.b_grad**2)
                weight_bias = self.bias_matrix - learning_rate*(self.b_grad/np.sqrt(velo_t_bias_rmsprop+1e-08))
                self.bias_matrix = weight_bias
                self.velocity_weight_bias_rmsprop = velo_t_bias_rmsprop
            self.t+=1
            if optimizer =='adam':
                #momentm logic
                velo_t_adam_h =0.9*self.m_weight_hidden_momentum_adam+(1-0.9)*self.W_hh_grad
                self.m_weight_hidden_momentum_adam = velo_t_adam_h
                velo_t_adam_xh = 0.9*self.m_weight_input_momentum_adam+(1-0.9)*self.W_xh_grad
                self.m_weight_input_momentum_adam = velo_t_adam_xh
                velo_t_adam_bias = 0.9*self.m_weight_bias_momentum_adam +(1-0.9)*self.b_grad
                self.m_weight_bias_momentum_adam = velo_t_adam_bias
                
                #velocity logic
                velo_t_v_adam_h =0.99*self.m_weight_hidden_adam+(1-0.99)*(self.W_hh_grad**2)
                self.m_weight_hidden_adam = velo_t_v_adam_h
                velo_t_v_adam_xh = 0.99*self.m_weight_input_adam+(1-0.99)*(self.W_xh_grad**2)
                self.m_weight_input_adam =  velo_t_v_adam_xh
                velo_t_v_bias_adam = 0.99*self.m_weight_bias_adam +(1-0.99)*(self.b_grad**2)
                self.m_weight_bias_adam = velo_t_v_bias_adam
                
                #bias correction
                m_weight_hidden_momentum_adam_corr = self.m_weight_hidden_momentum_adam/((1-(0.9**self.t)))
                m_weight_input_momentum_adam_corr =  self.m_weight_input_momentum_adam/((1-(0.9**self.t)))
                m_weight_bias_momentum_adam_corr = self.m_weight_bias_momentum_adam/((1-(0.9**self.t)))
                m_weight_hidden_adam_corr= self.m_weight_hidden_adam/((1-(0.99**self.t)))
                m_weight_input_adam_corr = self.m_weight_input_adam/((1-(0.99**self.t)))
                m_weight_bias_adam_corr = self.m_weight_bias_adam/((1-(0.99**self.t)))
                
                #weight update
                self.hidden_state_weight_matrix= self.hidden_state_weight_matrix- learning_rate *(m_weight_hidden_momentum_adam_corr/np.sqrt( m_weight_hidden_adam_corr+1e-08))
                self.weight_matrix = self.weight_matrix - learning_rate *(m_weight_input_momentum_adam_corr/np.sqrt( m_weight_input_adam_corr+1e-08))
                self.bias_matrix = self.bias_matrix - learning_rate *(m_weight_bias_momentum_adam_corr/np.sqrt( m_weight_bias_adam_corr+1e-08))
 
    def train(self, X, labels, batch_size, learning_rate=0.01, epochs=50, optimizer='sgd'):
        for epoch in range(epochs):
            for batch in range(0, X.shape[0], batch_size):
                X_batch = X[batch:batch+batch_size]
                labels_batch = labels[batch:batch+batch_size]

                # Forward through RNN
                final_hidden = self.rnn.forward(X_batch)

                # Forward through neural net head
                prediction = self.net.forward(final_hidden)

                # Backward through neural net head — returns the gradient at its input
                input_grad = self.net.backward(labels_batch, prediction)
                

                # Backward through RNN — uses input_grad as the starting incoming gradient
                # (this is where BPTT walks through timesteps; you'll write this part)
                self.rnn.BPTT(input_grad)
                
                # Apply optimizers — both networks update their weights
                self.net.optimizers(learning_rate, optimizer)
                self.rnn.rnn_optimizer(learning_rate, optimizer)
                    

                    

                    
                            
                        



                
                
                
                
                
                

            
        