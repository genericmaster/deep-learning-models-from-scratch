import torch as pt
device = pt.device('cuda')
from ENCODER import Encoder
from DECODER import DECODER
from TOKENIZER import InputEmbedding

import os
class Transformer(pt.nn.Module):
    def __init__(self, embedding_dim, heads, blocks, vocab_size_enc, vocab_size_dec, batch_size):
        super().__init__()
        self.encoder_embedding = InputEmbedding(vocab_size=vocab_size_enc, batch_size=batch_size, embedding_dim=embedding_dim)
        self.decoder_embedding = InputEmbedding(vocab_size=vocab_size_dec, batch_size=batch_size, embedding_dim=embedding_dim)
        self.Encoderblocks = Encoder(embedding_dim=embedding_dim, heads=heads, blocks=blocks)
        self.Decoderblocks = DECODER(embedding_dim=embedding_dim, heads=heads, blocks=blocks)
        self.linear = pt.nn.Linear(in_features=embedding_dim, out_features=vocab_size_dec)
        self.loss = pt.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, source_batch, target_batch,src_mask,trg_mask):
        # encoder
        encoder_embed = self.encoder_embedding.embedding(source_batch)
        encoder_matrix = self.encoder_embedding.positional_encoding(encoder_embed)
        encoder_context_matrix = self.Encoderblocks.encoder_forward(encoder_matrix,src_mask)

        # decoder
        decoder_embed = self.decoder_embedding.embedding(target_batch)
        decoder_matrix = self.decoder_embedding.positional_encoding(decoder_embed)
        decoder_context_matrix = self.Decoderblocks(decoder_matrix, encoder_context_matrix,src_mask,trg_mask)

        # linear
        logits = self.linear(decoder_context_matrix)
        return logits

    def fit(self, input_enc, input_dec, epochs, learning_rate):
        encoder_padding,src_mask = self.encoder_embedding.padding(input_enc)
        decoder_padding,trg_mask = self.decoder_embedding.padding(input_dec)
        optimizer = pt.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = pt.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) ** -0.5, (step + 1) * 4000 ** -1.5) * (4000 ** 0.5))
        loss_track = []
        start_epoch=0

        for epoch in range(start_epoch,epochs):

            for batch_enc, batch_dec, batch_src_mask, batch_trg_mask in zip(encoder_padding, decoder_padding, src_mask, trg_mask):
                batch_trg_mask = batch_trg_mask[:, :-1]  # match the decoder input shape
                decoder_input_batch = batch_dec[:, :-1]
                labels_batch = batch_dec[:, 1:]
                optimizer.zero_grad()
                logits = self.forward(batch_enc, decoder_input_batch,batch_src_mask,batch_trg_mask)
                logits = logits.permute(0, 2, 1)
                loss = self.loss(logits, labels_batch)
                loss_track.append(loss.item())
                loss.backward()  
                optimizer.step()                       
                scheduler.step()              
        return loss_track