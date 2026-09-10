import torch as pt
from TRANSFOMER_MODEL import Transformer

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=r'download and import tokenizer path from my hugginface account on readme')
device = pt.device('cuda')

model = Transformer(
    embedding_dim=256,
    heads =4,
    blocks=4,
    vocab_size_dec=32000,
    vocab_size_enc=32000,
    batch_size=1
).to(device)


weights = pt.load(r'download and import tokenizer path from my hugginface account on readme', map_location=device)
model.load_state_dict(weights)
model.eval()


def translate(model,sp,english_sentence,max_len=128,device=device):
    #tokenize input
     english_tokens=sp.encode(english_sentence,out_type=int,add_eos=True)
     english_tokens = pt.tensor(english_tokens, dtype=pt.long, device=device).unsqueeze(0)
   
    # adding a false mask because the tranformer expects some masking
     src_mask = pt.zeros(1, english_tokens.shape[1], dtype=pt.bool, device=device)
     enc_embed = model.encoder_embedding.embedding(english_tokens)
     enc_embed = model.encoder_embedding.positional_encoding(enc_embed)
     enc_out = model.Encoderblocks.encoder_forward(enc_embed, src_mask)
  

     #decoder section
     bos_id = sp.bos_id()
     eos_id = sp.eos_id()
     decoder_input = pt.tensor([[bos_id]], dtype=pt.long, device=device)
   
     for _ in range(max_len):
         trg_mask = pt.zeros(1, decoder_input.shape[1], dtype=pt.bool, device=device)
         dec_embed = model.decoder_embedding.embedding(decoder_input)
         dec_embed = model.decoder_embedding.positional_encoding(dec_embed)
         dec_out = model.Decoderblocks(dec_embed,enc_out,src_mask,trg_mask)
         logits = model.linear(dec_out)
         logits =logits[:, -1, :]
         next_token = pt.argmax(logits ,-1, keepdim=True)
         decoder_input = pt.cat([decoder_input,next_token],dim=1)
         if next_token == eos_id:
             break
     generated_ids = decoder_input.squeeze(0).tolist()[1:]  # skip <s>
     french = sp.decode(generated_ids)
     return french



input = input()
print(f"FR: {translate(model, sp, input)}")

