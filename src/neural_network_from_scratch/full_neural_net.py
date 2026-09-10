import numpy as np
import math 
np.random.seed(42)
f' this script handles both foward and backpropagation so that the model actually learns from data, added optimizers include sgd,momentum,rmsprop,adam'

class Neural_net:
   def __init__(self,loss):
      self.add_layer_list = [] 
      self.output_layer = None
      self.loss = loss
      self.incoming_gradient_list = []
      self.loss_track = []
      self.prediction_list = []
      self.loss_track_average =[]

   class Layer:
     def __init__(self,neurons,features,activation):
         self.neurons = neurons
         self.features = features
         self.activation = activation
         self.weight_matrix = self.weight_initialization()
         self.bias_matrix = self.bias_initialization()
         self.input = None
         self.z = None
         self.output = None
       
     def weight_init_choice(self):
         if self.activation in ['sigmoid', 'tanh','softmax']:
            weight_matrix = np.random.uniform(low= -math.sqrt(6/(self.features+self.neurons)),high=math.sqrt(6/(self.features+self.neurons)),size=(self.features,self.neurons))
         elif self.activation == "linear":
            weight_matrix = np.random.randn(self.features,self.neurons)
         elif self.activation =="relu":
            weight_matrix = np.random.uniform(low =-math.sqrt(2/self.features), high= math.sqrt(2/self.features),size=(self.features,self.neurons))  
         return weight_matrix
            
     def weight_initialization(self):
         return self.weight_init_choice()
             
     def bias_initialization(self):
            bias_matrix = np.zeros(shape=(1,self.neurons))
            return bias_matrix
         
     def forward(self,input):
         self.input = input
         linear_trans = np.dot(a=input,b=self.weight_matrix)
         linear_trans = linear_trans + self.bias_matrix
         self.z = linear_trans
         return self.forward_activation_choice(self.z)
      
     def forward_activation_choice(self,linear_trans):
                 if self.activation == "linear":
                     self.output = linear_trans     
                 if self.activation =="sigmoid":
                     self.output= 1/(1+np.exp(-linear_trans)) 
                 if self.activation =="tanh":
                     self.output =  (np.exp(linear_trans)-np.exp(-linear_trans))/(np.exp(linear_trans)+np.exp(-linear_trans))
                 if self.activation =="relu":
                     self.output = np.maximum(linear_trans,0)
                     
                 if self.activation =="softmax":
                                if self.neurons>1:
                                   max = np.max(linear_trans,axis=1,keepdims=True)
                                   softmax = np.exp(linear_trans-max)/np.sum(np.exp(linear_trans-max),axis=1, keepdims=True) #subtract max to avoid  number overflow
                                   self.output = softmax
                                  
                                else:
                                   raise ValueError("neurons must be set>1")
                 return self.output
      
     def backward_output_layer(self,labels,loss):
        if loss in ['mse','binary_cross_entropy','categorical_cross_entropy']:
           delta = (self.output-labels)/self.output.shape[0]
           gradient = self.input.T @ delta
           outgoing_gradient = delta @ self.weight_matrix.T
           return outgoing_gradient,gradient,delta
        
     def backward_prop(self,incoming_gradient):
        if self.activation == 'linear':
            delta = incoming_gradient * 1
            gradient = self.input.T @ delta
            outgoing_gradient = delta @ self.weight_matrix.T   
        if self.activation == "sigmoid":
            delta = incoming_gradient * (self.output * (1 - self.output))
            gradient = self.input.T @ delta
            outgoing_gradient = delta @ self.weight_matrix.T 
        if self.activation == 'tanh':
            delta = incoming_gradient*(1- (self.output)**2)
            gradient = self.input.T@delta
            outgoing_gradient =(delta @ self.weight_matrix.T)  
        if self.activation == 'relu':
            delta = incoming_gradient * np.where(self.z > 0, 1, 0)
            gradient = self.input.T @ delta
            outgoing_gradient = delta @ self.weight_matrix.T 
        else:
              outgoing_gradient =0
              delta =0
              gradient =0
        return outgoing_gradient ,gradient,delta         
                      
   def add_layer(self,neurons,features,activation):
      layer = Neural_net.Layer(neurons, features, activation)
      self.add_layer_list.append(layer)
          
   def forward_prop(self,user_data):
      current = user_data
      for layer in  self.add_layer_list:
        current=layer.forward(current)
      return current
     
   def loss_function(self,prediction,true_labels,loss_type='normal'):
      if loss_type == 'normal':
         if self.loss == 'mse':
            mse = (1/prediction.shape[0])*np.sum((prediction-true_labels)**2)
            return float(mse)
         if self.loss == 'binary_cross_entropy':
            if all(p in [0,1] for p in true_labels):
               binary_cross_entropy = -(1/prediction.shape[0]) * np.sum(true_labels*np.log(prediction) + (1-true_labels)*np.log(1-prediction))
            else:
               raise ValueError("binary_cross_entropy only requires inputs of 0 or 1")
            return float(binary_cross_entropy)
         if self.loss == "categorical_cross_entropy":
               categorical_cross_entropy = -(np.sum(np.sum(true_labels*np.log(np.clip(prediction, 1e-7, 1)),axis=1))/prediction.shape[0])
               return float(categorical_cross_entropy)
           
      elif loss_type == 'average':
            sum =0
            if np.array_equal(np.unique(true_labels), [0,1]):
                  for prediction,truth_label in zip(self.prediction_list,true_labels):
                           categorical_cross_entropy = -(np.sum(np.sum(truth_label*np.log(np.clip(prediction, 1e-7, 1)))))
                           sum+= float(categorical_cross_entropy)
            else:
               raise ValueError("categorical_cross_entropy only requires inputs of 0 or 1")
            average_loss = sum/len(self.prediction_list)
            self.loss_track_average.append(average_loss)
            return average_loss
         
   def forward(self,X):
        prediction = self.forward_prop(X)
        self.prediction_list.append(prediction)
        return prediction
     
   def backward(self,labels,prediction,loss_type):
               self.gradient_descent = []
               loss_value = self.loss_function(prediction,labels,loss_type)
               self.loss_track.append(loss_value)
               output_layer =1 
               for layer in reversed(self.add_layer_list):
                  if output_layer==1:
                     incoming_gradient,gradient,delta = layer.backward_output_layer(labels, self.loss)
                     output_delta = delta
                     output_gradient = gradient  
                  elif output_layer>1:
                     incoming_gradient,gradient,delta = layer.backward_prop(incoming_gradient)
          
                  self.incoming_gradient_list.append(incoming_gradient)
                  self.gradient_descent.append((layer, gradient,delta))
                  output_layer +=1
               self.output_gradient = output_gradient
               self.output_delta = output_delta
                  
               return incoming_gradient
      
   def optimizers(self,learning_rate,optimizer):
         gradient_descent = self.gradient_descent
         if not hasattr(self,'velocity_weight'):
            self.velocity_weight = [0]*len(self.add_layer_list)
            self.velocity_output_layer_weight=0
            self.velocity_output_layer_bias =0
            self.velocity_bias = [0]*len(self.add_layer_list)
            self.velocity_weight_rmsprop =[0]*len(self.add_layer_list)
            self.velocity_bias_rmsprop =  [0]*len(self.add_layer_list)
            self.velocity_output_layer_weight_rmsprop=0
            self.velocity_output_layer_bias_rmsprop=0
            self.velocity_weight_adam = [0]*len(self.add_layer_list)
            self.velocity_bias_adam =[0]*len(self.add_layer_list)
            self.m_weight_adam =[0]*len(self.add_layer_list)
            self.m_bias_adam =[0]*len(self.add_layer_list)
            self.m_weight_output_layer_adam =0
            self.m_bias_output_layer_adam =0
            self.velocity_output_layer_weight_adam =0
            self.velocity_output_layer_bias_adam =0
            self.t=0
            
         self.t+=1
         if optimizer =='sgd':   
                  for layer, gradient,delta in gradient_descent:
                     layer.weight_matrix = layer.weight_matrix - learning_rate * gradient
                     layer.bias_matrix = layer.bias_matrix - learning_rate * np.sum(delta, axis=0, keepdims=True)
                                                       
         if optimizer == 'rmsprop':
            for index, (gradient, v_t,v_t_bias) in enumerate(zip(gradient_descent, self.velocity_weight_rmsprop,self.velocity_bias_rmsprop)):
               v_t = 0.9*v_t +(1-0.9)*(gradient[1]**2)
               v_t_bias =0.9*v_t_bias +(1-0.9)* (np.sum(gradient[2]**2, axis=0, keepdims=True))
               gradient[0].weight_matrix=gradient[0].weight_matrix - learning_rate * (gradient[1]/np.sqrt(v_t+1e-08))
               gradient[0].bias_matrix = gradient[0].bias_matrix - learning_rate*(np.sum(gradient[2], axis=0, keepdims=True)/np.sqrt(v_t_bias+1e-08))
               self.velocity_weight_rmsprop[index] = v_t
               self.velocity_bias_rmsprop[index]=v_t_bias
               
         if optimizer == "momentum":
            for index, (gradient, v_t,v_t_bias) in enumerate(zip(gradient_descent, self.velocity_weight,self.velocity_bias)):
               v_t = 0.9*v_t +(1-0.9)*gradient[1]
               v_t_bias =0.9*v_t_bias +(1-0.9)* np.sum(gradient[2], axis=0, keepdims=True)
               gradient[0].weight_matrix=gradient[0].weight_matrix - learning_rate * v_t
               gradient[0].bias_matrix = gradient[0].bias_matrix - learning_rate*v_t_bias
               self.velocity_weight[index] = v_t
               self.velocity_bias[index]=v_t_bias
 
         if optimizer== 'adam':
         
            for index,(gradient,v_t_adam,v_t_adam_bias,m_t_adam,m_t_bias_adam) in enumerate (zip(gradient_descent,self.velocity_weight_adam,self.velocity_bias_adam,self.m_weight_adam,self.m_bias_adam)):
               #momentum
               m_t_adam = 0.9*m_t_adam +(1-0.9)*gradient[1]
               m_t_bias_adam =0.9*m_t_bias_adam +(1-0.9)* np.sum(gradient[2], axis=0, keepdims=True)
               self.m_weight_adam[index] = m_t_adam
               self.m_bias_adam[index]= m_t_bias_adam
               
               #rmsprop
               v_t_adam = 0.99*v_t_adam +(1-0.99)*(gradient[1]**2)
               v_t_adam_bias =0.99*v_t_adam_bias +(1-0.99)* (np.sum(gradient[2]**2, axis=0, keepdims=True))
               self.velocity_weight_adam[index] = v_t_adam
               self.velocity_bias_adam[index] = v_t_adam_bias
            #bias correction
               m_t_correction = m_t_adam/((1-(0.9**self.t)))
               m_t_bias_adam_correction = m_t_bias_adam/((1-(0.9**self.t)))
               v_t_adam_correction = v_t_adam/((1-(0.99**self.t)))
               v_t_adam_bias_correction = v_t_adam_bias/((1-(0.99**self.t)))
               
            #weight update
               gradient[0].weight_matrix=gradient[0].weight_matrix - learning_rate *(m_t_correction/np.sqrt(v_t_adam_correction+1e-08))
               gradient[0].bias_matrix = gradient[0].bias_matrix - learning_rate*(m_t_bias_adam_correction/np.sqrt(v_t_adam_bias_correction+1e-08))
                  
   def train(self, X, labels, batch_size,epoch, learning_rate=0.01, optimizer='sgd'):
    for i in range(epoch):
        for batch in range(0, X.shape[0] - X.shape[0] % batch_size ,batch_size):
            X_batch = X[batch:batch+batch_size]
            labels_batch = labels[batch:batch+batch_size]
            prediction = self.forward(X_batch)
            self.backward(labels_batch, prediction,loss_type='normal')
            self.optimizers(learning_rate, optimizer)
    
    return self.loss_track   
   