import sacrebleu
import json
from inference import translate,sp
from TRANSFOMER_MODEL import Transformer
import torch as pt

device = pt.device('cuda')

with open(r"src\transfomers_from_scratch\test_pairs.json", 'r',encoding='utf-8') as f:
    data = json.load(f)

test_en = data['en']
test_fr = data['fr']

# Change this for each model
weights_path = r"download and import model path from my hugginface account on readme" #swap for each variant
model = Transformer(embedding_dim=256, heads=4, blocks=4,
                    vocab_size_enc=32000, vocab_size_dec=32000,
                    batch_size=64).to(device)
weights = pt.load(weights_path, map_location=device)
model.load_state_dict(weights)
model.eval()

hypotheses = []
references = []
print('1')
with pt.no_grad():
    for i in range(2000):
        fr_hyp = translate(model, sp, test_en[i])
        hypotheses.append(fr_hyp)
        references.append(test_fr[i])
        if i % 100 == 0:
            print(f'{i}/2000', flush=True)

bleu = sacrebleu.corpus_bleu(hypotheses, [references])
print(f'BLEU: {bleu.score:.2f}')