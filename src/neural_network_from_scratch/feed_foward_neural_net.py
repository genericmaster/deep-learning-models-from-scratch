import numpy as np
import math 
 

#facade design pattern
class Feed_forward_net:
   def __init__(self,loss):
      self.add_layer_list = [] 
      self.output_layer = None
      self.loss = loss
 
   class Layer:
     def __init__(self,neurons,features,activation):
         self.neurons = neurons
         self.features = features
         self.activation = activation
         self.weight_matrix = self.weight_initialization()
         self.bias_matrix = self.bias_initialization()
         
     def weight_initialization(self):
          if self.activation in ['sigmoid', 'tanh','softmax']:
            weight_matrix = np.random.uniform(low= -math.sqrt(6/(self.features+self.neurons)),high=math.sqrt(6/(self.features+self.neurons)),size=(self.features,self.neurons))
          elif self.activation == "linear":
            weight_matrix = np.random.randn(self.features,self.neurons)
          elif self.activation =="relu":
            weight_matrix = np.random.uniform(low =-math.sqrt(2/self.features), high= math.sqrt(2/self.features),size=(self.features,self.neurons))
          return weight_matrix
     def bias_initialization(self):
            bias_matrix = np.zeros(shape=(1,self.neurons))
            return bias_matrix
    
          
     def forward(self,input):
         linear_trans = np.dot(a=input,b=self.weight_matrix)
         linear_trans = linear_trans + self.bias_matrix
         if self.activation == 'linear':
            return linear_trans
         if self.activation =='sigmoid':
            sigmoid = 1/(1+np.exp(-linear_trans) )
            return sigmoid
         if self.activation =="tanh":
            tanh =  (np.exp(linear_trans)-np.exp(-linear_trans))/(np.exp(linear_trans)+np.exp(-linear_trans))
            return tanh
         if self.activation =="relu":
            relu = np.maximum(linear_trans,0)
            return relu
     def output_layer_calc(self,input,activation):
         linear_trans = np.dot(a=input,b=self.weight_matrix)
         linear_trans = linear_trans + self.bias_matrix
         if activation in ['linear','sigmoid','softmax']:
            if activation== 'linear':
               return linear_trans
            if activation== 'sigmoid':
               sigmoid = 1/(1+np.exp(-linear_trans))
               return sigmoid
            if activation =="softmax":
               if self.neurons>1:
                  max = np.max(linear_trans,axis=1,keepdims=True)
                  softmax = np.exp(linear_trans-max)/np.sum(np.exp(linear_trans-max),axis=1, keepdims=True) #subtract max to avoid  number overflow
                  return softmax
               else:
                  raise ValueError("neurons must be set>1")
         else:
            raise ValueError("specify required activation from linear , sigmoid and softmax")
               
   def add_layer(self,neurons,features,activation):
      layer = Feed_forward_net.Layer(neurons, features, activation)
      self.add_layer_list.append(layer)
      
   def result_layer(self,neurons,features,activation):
      self.output_activation = activation
      self.output_layer = Feed_forward_net.Layer(neurons,features,self.output_activation)
               
   def foward_prop(self,user_data):
      current = user_data
      for layer in  self.add_layer_list:
        current=layer.forward(current)
      
      current = self.output_layer.output_layer_calc(current,self.output_activation)
      return current
     
   def loss_function(self,prediction,true_labels):
      if self.loss == 'rmse':
         rmse = np.sqrt((1/prediction.shape[0])*np.sum((prediction-true_labels)**2))
         return rmse
      if self.loss == 'binary_cross_entropy':
         if all(p in [0,1] for p in true_labels):
            binary_cross_entropy = 1/prediction.shape[0]*np.sum(-true_labels*np.log(prediction)+ (1-true_labels)*np.log(1-prediction))
         else:
            raise ValueError("binary_cross_entropy only requires inputs of 0 or 1")
         return binary_cross_entropy
      if self.loss == "categorical_cross_entropy":
          if np.array_equal(np.unique(true_labels), [0,1]):
            categorical_cross_entropy = -(np.sum(np.sum(true_labels*np.log(np.clip(prediction, 1e-7, 1)),axis=1))/prediction.shape[0])
            return categorical_cross_entropy
          else:
            raise ValueError("categorical_cross_entropy only requires inputs of 0 or 1")
         


         
         
          
      
            
      
         
      
            
         

            
         
      

      

         
      
      




