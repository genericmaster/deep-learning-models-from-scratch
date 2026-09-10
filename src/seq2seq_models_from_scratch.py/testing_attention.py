import torch as pt
import gzip
from seq2seq_using_attention import Attention

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

model = Attention(
    hidden_dim_enc=256, input_dim_enc=256, batch_size_enc=64,
    hidden_dim_dec=256, input_dim_dec=256, batch_size_dec=64,
    vocab_size=17005
)

checkpoint = pt.load(
    r'C:\THEDO DONT TOUCH\machine learning basics\attention_model.pt',
    map_location=device
)
model.encoder.hidden_weight = checkpoint['encoder_hidden_weight']
model.encoder.input_weight = checkpoint['encoder_input_weight']
model.encoder.bias = checkpoint['encoder_bias']
model.decoder.hidden_weight = checkpoint['decoder_hidden_weight']
model.decoder.input_weight = checkpoint['decoder_input_weight']
model.decoder.bias = checkpoint['decoder_bias']
model.decoder.net_encoder.load_state_dict(checkpoint['net_encoder'])
model.decoder.net_prev_hidden.load_state_dict(checkpoint['net_prev_hidden'])
model.decoder.net_pre_aplha.load_state_dict(checkpoint['net_pre_alpha'])
model.decoder.net_prediction.load_state_dict(checkpoint['net_prediction'])

english_tensors = pt.load('data/english_embedded.pt')
new_tensor = [batch.unsqueeze(0) for batch in english_tensors]

appended_list_fr = pt.load('data/french_tokens.pt')
tokens_fr = {word: index for index, word in enumerate(appended_list_fr)}
index_to_word = {index: word for word, index in tokens_fr.items()}

french_embedding_matrix = pt.load('data/french_embedding_matrix.pt').to(device)

sos_index = tokens_fr['<sos>']
sos_embedding = french_embedding_matrix[sos_index].unsqueeze(0)

model.encoder.batch_size = 1
model.decoder.batch_size = 1


def translate(sentence_tensor):
    enc_input = sentence_tensor.to(device)
    final_hidden, hidden_list = model.encoder.encoder_foward(enc_input)

    decoder_input = sos_embedding
    hidden = final_hidden[0].unsqueeze(0)
    words = []

    for _ in range(50):
        context = model.decoder.alignment_model(hidden_list, hidden)
        linear_trans = pt.matmul(decoder_input, model.decoder.input_weight.T) + \
                       pt.matmul(hidden, model.decoder.hidden_weight.T) + \
                       model.decoder.bias + context
        hidden = pt.tanh(linear_trans)
        prediction = model.decoder.net_prediction(hidden)
        index = prediction.argmax(dim=1).item()
        word = index_to_word[index]
        if word == '<eos>':
            break
        words.append(word)
        decoder_input = french_embedding_matrix[index].unsqueeze(0)

    return ' '.join(words)


with gzip.open(
    r'C:\THEDO DONT TOUCH\machine learning basics\data\multi30k\train.en.gz', 'rt'
) as f:
    english_sentences = f.read().splitlines()

with gzip.open(
    r'C:\THEDO DONT TOUCH\machine learning basics\data\multi30k\train.fr.gz',
    'rt', encoding='utf-8'
) as f:
    french_sentences = f.read().splitlines()

for i in range(5):
    print('English:', english_sentences[i])
    print('True French:', french_sentences[i])
    print('Predicted:', translate(new_tensor[i]))
    print()