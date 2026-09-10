import numpy as np
import math

np.random.seed(42)
class Conv_network:
    def  __init__(self):
         self.add_conv_layer_list = []
         
         
               
    class Conv_layer:
        def __init__(self,filter,kernel,depth,stride,padding,pool,pool_size,pool_stride):
            self.filter =filter
            self.kernel = kernel
            self.depth =depth
            self.stride = stride
            self.padding = padding
            self.z = None
            self.weight_matrix = self.kernel_weight_initialization()
            self.activation_map = None
            self.pool = pool
            self.pool_stride = pool_stride
            self.pool_size = pool_size
            
            
        def kernel_weight_initialization(self):
            kernel_weight = np.random.uniform(low =-math.sqrt(2/(self.kernel*self.kernel*self.depth)), high= math.sqrt(2/(self.kernel*self.kernel*self.depth)),size=(self.filter,self.depth,self.kernel,self.kernel))
            return kernel_weight
        
        def conv_layer_computation(self,input):
            output_spatial_size =  math.floor((((input.shape[1]+(2*self.padding))-self.kernel)/self.stride) +1)
            output_size = np.zeros((self.filter,output_spatial_size,output_spatial_size))
            kernel_weights = self.weight_matrix
            
            for filter in range(self.filter):
                for row in range(output_spatial_size):
                    row_start = row*self.stride
                    for column in range(output_spatial_size):
                        column_start =column*self.stride
                        linear_trans = np.sum(kernel_weights[filter]*input[:,row_start:row_start+self.kernel,column_start:column_start+self.kernel])
                        output_size[filter,row,column]=linear_trans
                        
            self.z =output_size   
            return output_size
                
        def activation(self):
            self.activation_map = np.maximum(self.z,0)
            return self.activation_map
        def pooling(self,input):

            grid = math.floor(((input.shape[1]-self.pool_size)/self.pool_stride) +1)
            output_size = np.zeros((input.shape[0],grid,grid))

            for filter in range(input.shape[0]):
                for row in range(grid):
                    row_start = row*self.pool_stride
                    for column in range(grid):
                        column_start = column*self.pool_stride
                        output = np.max(input[filter,row_start:row_start+self.pool_size,column_start:column_start+self.pool_size])
                        output_size [filter,row,column]= output
            return output_size

        
    def add_conv_layer(self,filter,kernel,depth,stride,padding,pool,pool_size,pool_stride):
        layer = Conv_network.Conv_layer(filter,kernel,depth,stride,padding,pool,pool_size,pool_stride)
        self.add_conv_layer_list.append(layer)
        
    def train(self,input):
        current =input
        full_array =[]
        for index in range(input.shape[0]):
            current = input[index]
            for layer in self.add_conv_layer_list:
                if layer.pool == False:
                    current = layer.conv_layer_computation(current)
                    current =layer.activation()
                elif layer.pool == True:
                    current = layer.conv_layer_computation(current)
                    current = layer.activation()
                    current = layer.pooling(current)
 
            current= current.flatten()
            full_array.append(current)
        current=np.vstack(full_array)
        return current
    
            
            
            
        
        
        
        
        
        
        
         
        
        
        
        
        
            
        
            
            
            
            
            
            