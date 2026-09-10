import torch as pt
device= pt.device('cuda')

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

    def forward(self, input,trg_mask):
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
        attention_score = attention_score.masked_fill(trg_mask[:, None, None, :], float('-inf'))
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


class MultiHeadCrossAttention(pt.nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.heads = heads
        self.embedding = embedding_dim
        self.head_dim = self.embedding // self.heads
        self.query = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.project = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, input_enc, input_dec,src_mask):
        sequence_length_dec = input_dec.shape[1]
        batch_size_dec = input_dec.shape[0]
        sequence_length_enc = input_enc.shape[1]
        batch_size_enc = input_enc.shape[0]
        query = self.query(input_dec)
        self.q = pt.reshape(query, shape=(batch_size_dec, sequence_length_dec, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        key = self.key(input_enc)
        self.k = pt.reshape(key, shape=(batch_size_enc, sequence_length_enc, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        value = self.value(input_enc)
        self.v = pt.reshape(value, shape=(batch_size_enc, sequence_length_enc, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        attention_score = (pt.matmul(self.q, self.k.permute(0, 1, 3, 2))) / (self.head_dim ** 0.5)
        attention_score = attention_score.masked_fill(src_mask[:, None, None, :], float('-inf'))
        attention_weights = pt.softmax(attention_score, dim=-1)
        context_vector = pt.matmul(attention_weights, self.v).permute(0, 2, 1, 3).contiguous().reshape(batch_size_dec, sequence_length_dec, self.embedding)
        output = self.project(context_vector)
        return output


class DecoderBlock(pt.nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()
        self.maskedatttention = MaskedMultiHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.layernorm1 = LayerNorm(embedding_dim=embedding_dim)
        self.crossattention = MultiHeadCrossAttention(embedding_dim=embedding_dim, heads=heads)
        self.feedfoward = FeedFoward(embedding_dim=embedding_dim)
        self.layernorm2 = LayerNorm(embedding_dim=embedding_dim)
        self.layernorm3 = LayerNorm(embedding_dim=embedding_dim)

    def dec_block_forward(self, input, enc_output,src_mask,trg_mask):
        context_vector = self.maskedatttention(input,trg_mask)
        add = context_vector + input
        normalization = self.layernorm1(add)
        cross_attention = self.crossattention(enc_output, normalization,src_mask)
        add2 = cross_attention + normalization
        normalization2 = self.layernorm2(add2)
        linear = self.feedfoward(normalization2)
        add3 = linear + normalization2
        normalization3 = self.layernorm3(add3)
        return normalization3


class DECODER(pt.nn.Module):
    def __init__(self, embedding_dim, heads, blocks):
        super().__init__()
        self.decoderblock_list = pt.nn.ModuleList([
            DecoderBlock(embedding_dim=embedding_dim, heads=heads)
            for _ in range(blocks)
        ])

    def forward(self, input, enc_output,src_mask,trg_mask):
        current = input
        for decoder in self.decoderblock_list:
            current = decoder.dec_block_forward(current, enc_output,src_mask,trg_mask)
        return current
    
    
   
        
    
    
    