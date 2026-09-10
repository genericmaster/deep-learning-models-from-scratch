import torch as pt
device = pt.device('cuda' if pt.cuda.is_available else 'cpu')

class MultiHeadAttention(pt.nn.Module):
    def __init__(self,heads,embedding_dim):
        super().__init__()
        self.heads = heads
        self.embedding = embedding_dim
        self.head_dim = self.embedding//self.heads
        self.query =pt.nn.Linear(in_features=embedding_dim , out_features=embedding_dim).to(device)
        self.key=pt.nn.Linear(in_features=embedding_dim , out_features=embedding_dim).to(device)
        self.value=pt.nn.Linear(in_features=embedding_dim , out_features=embedding_dim).to(device)
        self.projecion=pt.nn.Linear(in_features=embedding_dim , out_features=embedding_dim).to(device)
        
    def forward(self,input,src_mask):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        query = self.query(input)
        self.q = pt.reshape(query,shape=(batch_size,sequence_length,self.heads,self.head_dim)).permute(0,2,1,3).to(device)
        key = self.key(input)
        self.k = pt.reshape(key,shape=(batch_size,sequence_length,self.heads,self.head_dim)).permute(0,2,1,3).to(device)
        value= self.value(input)
        self.v = pt.reshape(value,shape=(batch_size,sequence_length,self.heads,self.head_dim)).permute(0,2,1,3).to(device)
        
        attention_score = (pt.matmul(self.q,self.k.permute(0,1,3,2)))/(self.head_dim**0.5)
        attention_score = attention_score.masked_fill(src_mask[:, None, None, :], float('-inf'))
        attention_weights = pt.softmax(attention_score,dim=-1).to(device)
        context_vector = pt.matmul(attention_weights,self.v).permute(0, 2, 1, 3).contiguous().reshape(batch_size, sequence_length, self.embedding).to(device) #ai
        output= self.projecion(context_vector)
        return output
        
               
class LayerNorm(pt.nn.Module):
    def __init__(self,embedding_dim):
       super().__init__()
       self.gamma = pt.nn.Parameter(pt.ones(embedding_dim, device=device))
       self.beta = pt.nn.Parameter(pt.zeros(embedding_dim, device=device))
    def forward(self,input):
        mean = input.mean(dim=-1,keepdim=True)
        std = input.std(dim=-1,keepdim=True)
        normalization = (input-mean)/(std+1e-08)
        normalized_embedding= (normalization*self.gamma )+self.beta
        return normalized_embedding
    
    
class FeedForward(pt.nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.layer1  = pt.nn.Linear(in_features=embedding_dim,out_features=4*embedding_dim).to(device)
        self.layer2 = pt.nn.Linear(in_features=4*embedding_dim,out_features=embedding_dim).to(device)
        
    def forward(self,input):
       first_layer= self.layer1(input)
       activation = pt.relu(first_layer).to(device)
       second_layer = self.layer2(activation)
       return second_layer
       

class EncoderBlock(pt.nn.Module):
    def __init__(self,embedding_dim,heads):
        super().__init__()
        self.selfattention = MultiHeadAttention(embedding_dim=embedding_dim,heads=heads)
        self.layernorm1 = LayerNorm(embedding_dim=embedding_dim)
        self.feedfoward = FeedForward(embedding_dim=embedding_dim)
        self.layernorm2 = LayerNorm(embedding_dim=embedding_dim)
        
    def forward(self,input,src_mask):
        attention_vectors = self.selfattention.forward(input,src_mask)
        add = input+attention_vectors
        normalization = self.layernorm1.forward(add)
        foward_output = self.feedfoward.forward(normalization)
        final_add = normalization+foward_output
        attention_output = self.layernorm2.forward(final_add)
        return attention_output
        
        
class Encoder(pt.nn.Module):
    def __init__(self,embedding_dim,heads,blocks):
        super().__init__()
        self.blocks = blocks
        self.encoderblock_list=pt.nn.ModuleList([EncoderBlock(embedding_dim=embedding_dim, heads=heads)
            for _ in range(blocks)
        ])
    def encoder_forward(self,input,src_mask):
        current = input
        for encoder in self.encoderblock_list:
            current = encoder.forward(current,src_mask)
        return current
            
enc = Encoder(embedding_dim=512, heads=8, blocks=6)
