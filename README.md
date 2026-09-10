# Deep Learning Models From Scratch

A collection of deep learning architectures implemented from first principles — starting with plain NumPy (manual forward pass, manual backprop, manual optimizers) and progressing to PyTorch-based Transformer architectures. Built to understand what's actually happening inside the models frameworks like PyTorch and TensorFlow abstract away.

## Why this repo exists

Most of these models could be built in a few lines with `torch.nn`. The point here was the opposite: derive and implement the forward pass, the backward pass, and the optimizer update rules by hand, then confirm the model actually learns. Everything through the RNN/seq2seq stage uses hand-written backpropagation (no autograd); the Transformer and decoder-only models switch to PyTorch's autograd, since the focus at that stage shifted to architecture (attention, masking, positional encoding) rather than re-deriving the chain rule for a 6th time.

## What's in here

### 1. Feedforward Neural Network (`src/neural_network_from_scratch/`)
A from-scratch fully connected network in NumPy.
- `feed_foward_neural_net.py` — forward-pass-only version (early iteration)
- `full_neural_net.py` — complete network with manual backpropagation and four optimizers implemented from scratch: **SGD, Momentum, RMSProp, and Adam** (with bias correction)
- Supports `linear`, `sigmoid`, `tanh`, `relu`, and `softmax` activations, with activation-appropriate weight initialization (Xavier/Glorot for sigmoid/tanh, He for ReLU)
- Supports `mse`, `binary_cross_entropy`, and `categorical_cross_entropy` losses
- `testing.py` — trains the network on a Chicago taxi fare dataset (regression) and a synthetic multilabel classification dataset, comparing all four optimizers on convergence speed

### 2. Recurrent Neural Network (`src/recurrent neural networks from scratch/`)
A vanilla RNN (tanh activation) with full **Backpropagation Through Time (BPTT)**, implemented in NumPy, feeding into the feedforward network above as a classification head. Same four optimizers as above, adapted for recurrent weight sharing across timesteps.

### 3. Sequence-to-Sequence Models (`src/seq2seq_models_from_scratch.py/`)
*(Note: this is a directory, not a file — a naming leftover from scaffolding worth fixing.)*

Three progressively more complex seq2seq implementations, applied to English→French translation:
- `encoder.py` / `decoder.py` / `seq2seq.py` — a from-scratch NumPy encoder-decoder RNN with manual BPTT through both halves
- `seq2seq_using_attention.py` — a **Bahdanau-style additive attention** seq2seq model, hand-rolled with raw PyTorch tensors and manual weight matrices (relies on autograd for gradients, not manual backprop)
-for refrence bahdanau was the first researcher to propose attention as a method for improving seq models heres the link below if your interested in the paper
- `shonisa.py` — data preprocessing for the Multi30k EN-FR dataset (tokenization, random-init word embeddings, tensor serialization)
- `testing_attention.py` — loads a trained attention checkpoint and runs translation on sample sentences

### 4. Transformer (Encoder-Decoder) (`src/transfomers_from_scratch/`)
A full "Attention Is All You Need"-style Transformer in PyTorch, used for English→French machine translation:
- `ENCODER.py` / `DECODER.py` — multi-head self-attention, masked self-attention, encoder-decoder cross-attention, layer norm, and position-wise feedforward blocks, all implemented manually (not `torch.nn.MultiheadAttention`)
- `TOKENIZER.py` — word-level tokenizer with padding and mask generation, plus sinusoidal positional encoding
- `TRANSFOMER_MODEL.py` — assembles the full model, with a training loop using AdamW, gradient clipping, and a warmup + inverse-sqrt learning rate schedule
- `inference.py` — greedy-decoding translation from English to French
- `blue_scores.py` — evaluates translation quality with **BLEU score** (via `sacrebleu`) against a 2,000-sentence-pair test set (`test_pairs.json`)

### 5. Decoder-Only Model (`src/decoder_coding_model_from_scratch/`)
A GPT-style, decoder-only Transformer trained as a **code-completion model**, using SentencePiece BPE tokenization:
- `DECODER.py` — masked multi-head self-attention decoder blocks, pre-LN architecture, trained with a cosine learning-rate schedule and gradient clipping
- `greedydecoding_inference.py` — supports both standard autoregressive generation and **Fill-in-the-Middle (FIM)** inference (prefix/suffix/middle, in the style of Codex/StarCoder), with a repetition penalty applied during decoding

## Tech stack
- **NumPy** — manual forward/backward passes for the feedforward net, RNN, and base seq2seq model
- **PyTorch** — autograd-based training for the attention seq2seq, Transformer, and decoder-only models
- **SentencePiece** — BPE tokenization for the code model
- **sacrebleu** — translation quality evaluation
- **scikit-learn, pandas, matplotlib, seaborn** — data prep and visualization in the optimizer comparison notebook/script

## Known rough edges

This is a learning project, and a few things are still mid-cleanup:
- Several scripts contain hardcoded local file paths (Windows-style, e.g. `C:\...`) for datasets and model checkpoints — these need to be parameterized or replaced with relative paths / config before anyone else can run the code directly.
- `TRANSFOMER_MODEL.py` references a `global_step` variable that isn't initialized before use.
- `seq2seq_using_attention.py` saves a checkpoint before the training loop actually runs — the save call needs to move to the end of `train()`.
- The `pyproject.toml` doesn't yet declare dependencies (numpy, torch, pandas, scikit-learn, sentencepiece, sacrebleu, matplotlib, seaborn) — currently assumes they're installed separately.
- The `src/seq2seq_models_from_scratch.py` folder name should be renamed to drop the `.py` (it's a package, not a file).

## Status
Actively evolving as a personal learning project — architectures are implemented and trainable, but not yet packaged for one-command reproducibility.