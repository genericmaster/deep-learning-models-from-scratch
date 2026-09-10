import torch as pt
import math

device = pt.device('cuda')



class InputEmbedding(pt.nn.Module):
    def __init__(self,vocab_size,batch_size,embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedded_batch = None
        self.dict_embedding = pt.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim).to(device)
        self.input_embedding =None
        
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


class MaskedMultiHeadAttention(pt.nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()
        self.heads = heads
        self.embedding = embedding_dim
        self.head_dim = self.embedding // self.heads
        self.query = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.projection = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        query = self.query(input)
        self.q = pt.reshape(query, shape=(batch_size, sequence_length, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        key = self.key(input)
        self.k = pt.reshape(key, shape=(batch_size, sequence_length, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        value = self.value(input)
        self.v = pt.reshape(value, shape=(batch_size, sequence_length, self.heads, self.head_dim)).permute(0, 2, 1, 3)

        attention_score = (pt.matmul(self.q, self.k.permute(0, 1, 3, 2))) / (self.head_dim ** 0.5)
        masked_matrix = pt.triu(pt.ones(sequence_length, sequence_length, dtype=pt.bool), diagonal=1).to(device)
        attention_score = attention_score.masked_fill(masked_matrix, float('-inf'))
        attention_weights = pt.softmax(attention_score, dim=-1)
        context_vector = pt.matmul(attention_weights, self.v).permute(0, 2, 1, 3).contiguous().reshape(batch_size, sequence_length, self.embedding)
        output = self.projection(context_vector)
        return output



class LayerNorm(pt.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gamma = pt.nn.Parameter(pt.ones(embedding_dim))
        self.beta = pt.nn.Parameter(pt.zeros(embedding_dim))

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        normalization = (input - mean) / (std + 1e-08)
        normalized_embedding = (normalization * self.gamma) + self.beta
        return normalized_embedding



class FeedFoward(pt.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layer1 = pt.nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim)
        self.layer2 = pt.nn.Linear(in_features=4 * embedding_dim, out_features=embedding_dim)

    def forward(self, input):
        first_layer = self.layer1(input)
        activation = pt.relu(first_layer)
        second_layer = self.layer2(activation)
        return second_layer
    
    

#main purpose is to be an insatiation of a layer within the network
class DecoderBlock(pt.nn.Module):
    def __init__(self,heads,embedding_dim):
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(heads=heads,embedding_dim=embedding_dim)
        self.layer_norm_1 = LayerNorm(embedding_dim=embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim=embedding_dim)
        self.feed_foward = FeedFoward(embedding_dim=embedding_dim)
        
    def forward(self,x):
         # Pre-LN attention
        norm1 = self.layer_norm_1(x)
        attention_output = self.self_attention(norm1)
        residual = attention_output + x
        # Pre-LN feedforward
        norm2 = self.layer_norm_2(residual)
        mlp_output = self.feed_foward(norm2)
        output = residual + mlp_output
        return output
        
class Decoder(pt.nn.Module):
    def __init__(self,heads,blocks,embedding_dim):
        super().__init__()
        self.decoder_blocks_list = pt.nn.ModuleList([ DecoderBlock(embedding_dim=embedding_dim, heads=heads) for _ in range(blocks)])
        
    def forward(self,x):
        current = x
        for  decoder in self.decoder_blocks_list:
            current = decoder.forward(current)
        return current
        
        
        

class CodingModel(pt.nn.Module):
    def __init__(self, vocab_size,batch_size, embedding_dim, heads, blocks):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size=vocab_size,batch_size=batch_size, embedding_dim=embedding_dim)
        self.decoder_blocks = Decoder(heads=heads, blocks=blocks, embedding_dim=embedding_dim)
        self.linear = pt.nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        self.loss = pt.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, x):
        embeddings = self.input_embedding.embedding(x)
        positional_embed = self.input_embedding.positional_encoding(embeddings)
        decoder_context_matrix = self.decoder_blocks.forward(positional_embed)
        logits = self.linear(decoder_context_matrix)
        return logits

    def fit(self, chunk_files, epochs, batch_size, learning_rate):
        optimizer = pt.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.1)
        total_steps = 244000
        warmup_steps = 2000

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = pt.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        loss_track = []
        start_epoch = 0
        batch_count = 0

        for epoch in range(start_epoch, epochs):
            for file_path in chunk_files:
                chunks = pt.load(file_path, weights_only=False)
                chunks = [pt.tensor(c, dtype=pt.long) if not isinstance(c, pt.Tensor) else c for c in chunks]

                for i in range(0, len(chunks) - batch_size, batch_size):
                    batch = pt.stack(chunks[i:i+batch_size]).to(device)
                    decoder_input_batch = batch[:, :-1]
                    labels_batch = batch[:, 1:]
                    optimizer.zero_grad()

                
                    logits = self.forward(decoder_input_batch)
                    logits = logits.permute(0, 2, 1)
                    loss = self.loss(logits, labels_batch)

                    loss_track.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()        
        return loss_track
