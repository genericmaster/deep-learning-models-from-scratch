import gzip as reader
import numpy as np
import torch as pt

with reader.open(filename=r"C:\THEDO DONT TOUCH\machine learning basics\data\multi30k\train.en.gz",mode='rt')as file:
    dataset = file.read()
    word_list = dataset.split()
    sentence_full_list =dataset.splitlines()

unique_list = list(dict.fromkeys(word_list))
appended_list = ['<sos>']+unique_list + ['<eos>']
tokens={}
for index, word in enumerate(appended_list):
    tokens[word] = index

with reader.open(filename=r"C:\THEDO DONT TOUCH\machine learning basics\data\multi30k\train.fr.gz",mode='rt',encoding='utf-8')as file:
    dataset_fr = file.read()
    word_list_french = dataset_fr.split()
    sentence_full_list_fr =dataset_fr.splitlines()

unique_list_fr = list(dict.fromkeys(word_list_french))
appended_list_fr = ['<sos>']+unique_list_fr + ['<eos>']
tokens_fr={}
for index, word in enumerate(appended_list_fr):
    tokens_fr[word] = index

word_embbedding_english = np.random.normal(loc=0, scale=0.01,size=(len(tokens),256))
word_embbedding_french = np.random.normal(loc=0, scale=0.01,size=(len(tokens_fr),256))

def sentence_embedding(sentence_list,tokens,embeddings):
    final_matrix_list = []
    for sentence in sentence_list:
        if sentence: 
            word_vector_list =[]
            word_list=sentence.split()
            for word in word_list:
                token=tokens[word]
                word_vector_list.append(embeddings[token])
            sos =embeddings[tokens['<sos>']]
            eos=embeddings[tokens['<eos>']]
            final_matrix_list.append(np.vstack([sos,word_vector_list,eos]))
        else:
            print('null sentences exist')
    return final_matrix_list

def prediction_indices(sentence_list, tokens):
    final_index_list = []
    for sentence in sentence_list:
        if sentence:
            word_list = sentence.split()
            indices = [tokens[word] for word in word_list]
            final_index_list.append(pt.tensor(indices, dtype=pt.long))
    return final_index_list

english_embedded = sentence_embedding(sentence_full_list, tokens, word_embbedding_english)
french_embedded = sentence_embedding(sentence_full_list_fr, tokens_fr, word_embbedding_french)
french_label_indices = prediction_indices(sentence_full_list_fr, tokens_fr)

english_tensors = [pt.tensor(x, dtype=pt.float32) for x in english_embedded]
french_tensors = [pt.tensor(x, dtype=pt.float32) for x in french_embedded]

pt.save(english_tensors, 'data/english_embedded.pt')
pt.save(french_tensors, 'data/french_embedded.pt')
pt.save(french_label_indices, 'data/french_labels.pt')
pt.save(pt.tensor(word_embbedding_english, dtype=pt.float32), 'data/english_embedding_matrix.pt')
pt.save(pt.tensor(word_embbedding_french, dtype=pt.float32), 'data/french_embedding_matrix.pt')
pt.save(appended_list_fr, 'data/french_tokens.pt')

print(f'English vocab size: {len(tokens)}')
print(f'French vocab size: {len(tokens_fr)}')
print('saved')