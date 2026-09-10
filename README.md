# Deep Learning Models From Scratch

A collection of deep learning architectures implemented from first principles — starting with plain NumPy (manual forward pass, manual backprop, manual optimizers) and progressing to PyTorch-based Transformer architectures. Built to understand what's actually happening inside the models frameworks that  PyTorch and TensorFlow abstract away.

## Tech stack
- **NumPy** — feedforward net, RNN, and base seq2seq model
- **PyTorch** — attention seq2seq, Transformer, and decoder-only models
- **SentencePiece** — BPE tokenization for the code model
- **sacrebleu** — translation quality evaluation
- **scikit-learn, pandas, matplotlib, seaborn** — data prep and visualization

## Why this repo exists

Most of these models could be built in a few lines with `torch.nn`. The point here was the opposite: derive and implement the forward pass, the backward pass, and the optimizer update rules by hand, then confirm the model actually learns. As the architectures got more complex, the focus shifted from re-deriving the math to understanding the architecture itself — attention, masking, positional encoding.

## What's in here

### 1. Feedforward Neural Network (`src/neural_network_from_scratch/`)
A fully connected network in NumPy.
- `feed_foward_neural_net.py` — forward-pass-only version (early iteration)
- `full_neural_net.py` — complete network with backpropagation and four optimizers implemented from scratch: **SGD, Momentum, RMSProp, and Adam** (with bias correction)
- Supports `linear`, `sigmoid`, `tanh`, `relu`, and `softmax` activations, with activation-appropriate weight initialization
- Supports `mse`, `binary_cross_entropy`, and `categorical_cross_entropy` losses
- `testing.py` — trains the network on a Chicago taxi fare dataset (regression) and a synthetic multilabel classification dataset, comparing all four optimizers on convergence speed

### 2. Recurrent Neural Network (`src/recurrent neural networks from scratch/`)
A vanilla RNN (tanh activation) with full **Backpropagation Through Time (BPTT)**, feeding into the feedforward network above as a classification head. Same four optimizers as above, adapted for recurrent weight sharing across timesteps.

### 3. Sequence-to-Sequence Models (`src/seq2seq_models_from_scratch.py/`)
Three progressively more complex seq2seq implementations, applied to English→French translation:
- `encoder.py` / `decoder.py` / `seq2seq.py` — an encoder-decoder RNN with BPTT through both halves
- `seq2seq_using_attention.py` — a **Bahdanau-style additive attention** seq2seq model ([Bahdanau, Cho & Bengio, 2015](https://arxiv.org/abs/1409.0473)) — the paper that introduced attention as a way for the decoder to "look back" at relevant parts of the source sentence instead of relying on a single fixed-length context vector
- `shonisa.py` — data preprocessing for the Multi30k EN-FR dataset (tokenization, word embeddings, tensor serialization)
- `testing_attention.py` — loads a trained attention checkpoint and runs translation on sample sentences

### 4. Transformer (Encoder-Decoder) (`src/transfomers_from_scratch/`)
A full Transformer following [Vaswani et al., 2017 — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762), used for English→French machine translation:
- `ENCODER.py` / `DECODER.py` — multi-head self-attention, masked self-attention, encoder-decoder cross-attention, layer norm, and position-wise feedforward blocks
- `TOKENIZER.py` — word-level tokenizer with padding, masking, and sinusoidal positional encoding
- `TRANSFOMER_MODEL.py` — assembles the full model, with a training loop using AdamW, gradient clipping, and a warmup + inverse-sqrt learning rate schedule
- `inference.py` — greedy-decoding translation from English to French
- `blue_scores.py` — evaluates translation quality with **BLEU score** (via `sacrebleu`) against a 2,000-sentence-pair test set got a score of 21.28 which implies it can produce coherent sounding french from english input

### 5. Decoder-Only Model (`src/decoder_coding_model_from_scratch/`)
A GPT-style, decoder-only Transformer trained as a **code-completion model**, using SentencePiece BPE tokenization:
- `DECODER.py` — masked multi-head self-attention decoder blocks, pre-LN architecture, trained with a cosine learning-rate schedule and gradient clipping
- `greedydecoding_inference.py` — supports standard autoregressive generation as well as **Fill-in-the-Middle (FIM)** inference — prefix/suffix/middle generation, following [Bavarian et al., 2022 — "Efficient Training of Language Models to Fill in the Middle"](https://arxiv.org/abs/2207.14255) — with a repetition penalty applied during decoding

## Next steps
-  a shift t
- i plan to do write ups on all this projects detailing my struggle key things i learnt and more 

- comprehensive number of datasets will be chosen later on to test and bencjmark with actual frameworks like pytorch and tensorflow

- reproducing some papers like attention is all you need and bahmadu papers was really exciting  i reaaly got a feel for how this models actually learn  and the beuty in the simplicity of some of this algorithms 
