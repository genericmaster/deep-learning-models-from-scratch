import torch as pt
import gzip
import string 
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

def tokenizer(path):
   with gzip.open(filename=path,mode='rt',encoding='utf-8') as file:
         dataset=file.read().translate(str.maketrans("", "", string.punctuation))
         sentence_list = dataset.splitlines()
         word_list = dataset.split()
         word_list = list(dict.fromkeys(word_list))
         
   #tokenizing whole dataset
   tokens ={'<sos>':len(word_list)+1,'<eos>':len(word_list)+2,'<pad>':0}    
   for index,word in  enumerate(word_list,start=1):
       
       tokens[word]= index
       
 #tokenizing sentences
   tokenized_sentences =[]
   for sentence in sentence_list:
       word_list = sentence.split()
       word_list = ['<sos>']+word_list+['<eos>']
       token_list =[] 
       for word in word_list:
           token= tokens[word]
           token_list.append(token)  
       token_list=pt.tensor(token_list)#ai
       tokenized_sentences.append(token_list)
    
   return tokenized_sentences ,tokens
  

class InputEmbedding(pt.nn.Module):
    def __init__(self,vocab_size,batch_size,embedding_dim):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.batch_list = []
        self.mask_list=[]
        self.embedded_batch = None
        self.dict_embedding = pt.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim).to(device)
        self.input_embedding =None
    def padding(self,input):
        self.batch_list =[]
        self.mask_list=[]
        for batch in range(0,len(input),self.batch_size):
            chunk = input[batch:batch+self.batch_size]
            padded = pt.nn.utils.rnn.pad_sequence(chunk, batch_first=True, padding_value=0).to(device)
            mask = (padded == 0)
            self.batch_list.append(padded)
            self.mask_list.append(mask)
        return self.batch_list,self.mask_list
            
    def embedding(self,batch_matrix):
        embeddings= self.dict_embedding
        self.embedded_batch = embeddings(batch_matrix)    
        return self.embedded_batch
    def positional_encoding(self,batch_embedding):
        postional_matrix = pt.zeros_like(batch_embedding).to(device)
        intermediate_calc = batch_embedding.shape[1]
        position_sin = pt.arange(intermediate_calc).unsqueeze(1).to(device)
        positions_cos = pt.arange(intermediate_calc).unsqueeze(1).to(device)
        intermediate_calc2= self.embedding_dim//2
        sin_embedding_dim = pt.arange(intermediate_calc2).to(device)
        cos_embedding_dim = pt.arange(intermediate_calc2).to(device)
        
        postional_matrix[:,:,0:intermediate_calc2]= pt.sin(position_sin/1000**((2*sin_embedding_dim)/self.embedding_dim)).to(device)
        postional_matrix[:,:,intermediate_calc2:]= pt.cos(positions_cos/1000**((2*cos_embedding_dim)/self.embedding_dim)).to(device)
        
        self.input_embedding = batch_embedding+postional_matrix
        
        return self.input_embedding

         
         
        
        
        