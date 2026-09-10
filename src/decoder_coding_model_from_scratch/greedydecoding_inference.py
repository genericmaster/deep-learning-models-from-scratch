import sentencepiece as spm
import torch as pt
from DECODER import CodingModel

sp = spm.SentencePieceProcessor(model_file=r"download tokenizer from coding model folder thencopy path here")
device = pt.device('cuda')

model = CodingModel(vocab_size=32000, batch_size=16, embedding_dim=512, heads=8, blocks=12).to(device)
checkpoint = pt.load(r"download model weights from coding model folder then copy path here", map_location=device, weights_only=False)
weights = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
model.load_state_dict(weights)
model.eval()

EOT_ID = sp.piece_to_id('<EOT>')

def FIMInference(sp, model, prefix, suffix, max_length, temperature):
    model.eval()
    with pt.no_grad():
        prefix_tokens = [sp.piece_to_id('<PRE>')] + sp.encode(prefix)
        suffix_tokens = [sp.piece_to_id('<SUF>')] + sp.encode(suffix)
        decoder_input = prefix_tokens + suffix_tokens + [sp.piece_to_id('<MID>')]
        decoder_input = pt.tensor(decoder_input, dtype=pt.long, device=device).unsqueeze(0)
        for i in range(max_length):
            logits = model.forward(decoder_input)
            logits = logits[:, -1, :]
            logits = logits / temperature
            for token_id in set(decoder_input.squeeze(0).tolist()[-30:]):
                if token_id not in [2, 3, 4, 5]:
                    logits[0, token_id] = logits[0, token_id] / 1.5
            next_token = pt.argmax(logits, -1, keepdim=True)
            decoder_input = pt.cat([decoder_input, next_token], dim=1)
            if next_token.item() == EOT_ID:
                break
        prompt_length = len(prefix_tokens + suffix_tokens) + 1
        generated_tokens = decoder_input.squeeze(0).tolist()[prompt_length:]
        if generated_tokens and generated_tokens[-1] == EOT_ID:
            generated_tokens = generated_tokens[:-1]
        return sp.decode(generated_tokens), len(generated_tokens)
    
def ARInference(sp, model, prompt, max_length=100, temperature=0.5):
    model.eval()
    with pt.no_grad():
        decoder_input = sp.encode(prompt)
        decoder_input = pt.tensor(decoder_input, dtype=pt.long, device=device).unsqueeze(0)
        EOT_ID = sp.piece_to_id('<EOT>')
        for i in range(max_length):
            logits = model.forward(decoder_input)
            logits = logits[:, -1, :]
            logits = logits / temperature
            for token_id in set(decoder_input.squeeze(0).tolist()[-30:]):
                if token_id not in [2, 3, 4, 5]:
                    logits[0, token_id] = logits[0, token_id] / 1.5
            next_token = pt.argmax(logits, -1, keepdim=True)
            decoder_input = pt.cat([decoder_input, next_token], dim=1)
            if next_token.item() == EOT_ID:
                break
        prompt_length = len(sp.encode(prompt))
        generated_tokens = decoder_input.squeeze(0).tolist()[prompt_length:]
        if generated_tokens and generated_tokens[-1] == EOT_ID:
            generated_tokens = generated_tokens[:-1]
        return sp.decode(generated_tokens), len(generated_tokens)

print("=== AR COMPARISON: OOP DOCSTRINGS vs SIMPLE WITH DOCSTRINGS ===\n")

oop_with_docs = [
   """def name(y:list):
     square = [x*x fo x in y]
   """
        

]

for prompt in oop_with_docs:
    output, length = ARInference(sp, model, prompt, max_length=600, temperature=0.5)


print(output,length)