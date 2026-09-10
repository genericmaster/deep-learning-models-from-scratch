# Deep Learning Models From Scratch

A collection of deep learning architectures implemented from first principles — starting with plain NumPy (manual forward pass, manual backprop, manual optimizers) and progressing to PyTorch-based Transformer architectures. Built to understand what's actually happening inside the models frameworks that PyTorch and TensorFlow abstract away.

## Tech stack
- **NumPy** — feedforward net, CNN, RNN, and base seq2seq model
- **PyTorch** — attention seq2seq, Transformer, and decoder-only models
- **SentencePiece** — BPE tokenization for the code model
- **sacrebleu** — translation quality evaluation
- **Weights & Biases** — training/experiment monitoring for the Transformer models
- **scikit-learn, pandas, matplotlib, seaborn** — data prep and visualization

## Why this repo exists

Most of these models could be built in a few lines with `torch.nn`. The point here was the opposite: derive and implement the forward pass, the backward pass, and the optimizer update rules by hand, then confirm the model actually learns. As the architectures got more complex, the focus shifted from re-deriving the math to understanding the architecture itself — attention, masking, positional encoding.

The from-scratch feedforward network isn't just its own module — it's reused as the trainable head for the CNN, the RNN, and the base seq2seq model, so those are built end-to-end from the same components rather than each having its own separate dense layer implementation.

The move to PyTorch for the Transformer and decoder-only models wasn't just about architecture complexity — the models themselves got large enough that CPU training was no longer practical, and GPU support meant leaning on PyTorch.

## What's in here

### 1. Feedforward Neural Network (`src/neural_network_from_scratch/`)
A fully connected network in NumPy.
- `feed_foward_neural_net.py` — forward-pass-only version (early iteration)
- `full_neural_net.py` — complete network with backpropagation and four optimizers implemented from scratch: **SGD, Momentum, RMSProp, and Adam** (with bias correction)
- Supports `linear`, `sigmoid`, `tanh`, `relu`, and `softmax` activations, with activation-appropriate weight initialization
- Supports `mse`, `binary_cross_entropy`, and `categorical_cross_entropy` losses
- `testing.py` — trains the network on a Chicago taxi fare dataset (regression) and a synthetic multilabel classification dataset, comparing all four optimizers on convergence speed

### 2. Convolutional Neural Network (`src/convolutional_neural_network_from_scratch/`)
A CNN built in NumPy, benchmarked against an architecture-matched PyTorch model on FashionMNIST.
- `convolutional_network_from_scratch.py` — manual convolution (sliding-window), ReLU activation, and max pooling
- `convolution_network_using_pytorch.py` — sanity-checks the from-scratch convolution by applying a known edge-detection kernel and comparing feature maps against PyTorch's `Conv2d`
- `classification_on_networks_built_from_scratch.py` — feeds the from-scratch conv layers into the from-scratch feedforward network (with Adam) to classify FashionMNIST. The two halves weren't trained end-to-end here — one side's weights are frozen while the other trains — since mixing hand-rolled and framework components this way doesn't give a clean path to backprop the gradient across both
- `fully_using_pytorch.py` — the same architecture trained fully in PyTorch, as a head-to-head benchmark for accuracy and training behaviour
- `scratch_network_visualization.py` — visualizes predictions and intermediate feature maps from the from-scratch network

### 3. Recurrent Neural Network (`src/recurrent neural networks from scratch/`)
A vanilla RNN (tanh activation) with full **Backpropagation Through Time (BPTT)**, feeding into the feedforward network above as a classification head. Same four optimizers as above, adapted for recurrent weight sharing across timesteps.

### 4. Sequence-to-Sequence Models (`src/seq2seq_models_from_scratch.py/`)
Three progressively more complex seq2seq implementations, applied to English→French translation:
- `encoder.py` / `decoder.py` / `seq2seq.py` — an encoder-decoder RNN with BPTT through both halves, again using the from-scratch feedforward network as the decoder's prediction head
- `seq2seq_using_attention.py` — a **Bahdanau-style additive attention** seq2seq model ([Bahdanau, Cho & Bengio, 2015](https://arxiv.org/abs/1409.0473)) — the paper that introduced attention as a way for the decoder to "look back" at relevant parts of the source sentence instead of relying on a single fixed-length context vector
- `shonisa.py` — data preprocessing for the Multi30k EN-FR dataset (tokenization, word embeddings, tensor serialization)
- `testing_attention.py` — loads a trained attention checkpoint and runs translation on sample sentences

### 5. Transformer (Encoder-Decoder) (`src/transfomers_from_scratch/`)
A full Transformer following [Vaswani et al., 2017 — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762), used for English→French machine translation:
- `ENCODER.py` / `DECODER.py` — multi-head self-attention, masked self-attention, encoder-decoder cross-attention, layer norm, and position-wise feedforward blocks
- `TOKENIZER.py` — word-level tokenizer with padding, masking, and sinusoidal positional encoding
- `TRANSFOMER_MODEL.py` — assembles the full model, with a training loop using AdamW, gradient clipping, and a warmup + inverse-sqrt learning rate schedule, monitored with Weights & Biases
- `inference.py` — greedy-decoding translation from English to French
- `blue_scores.py` — evaluates translation quality with **BLEU score** (via `sacrebleu`) against a 2,000-sentence-pair test set — scored 21.28, producing coherent French from English input

### 6. Decoder-Only Model (`src/decoder_coding_model_from_scratch/`)
A GPT-style, decoder-only Transformer trained as a **code-completion model**, using SentencePiece BPE tokenization:
- `DECODER.py` — masked multi-head self-attention decoder blocks, pre-LN architecture, trained with a cosine learning-rate schedule and gradient clipping
- `greedydecoding_inference.py` — supports standard autoregressive generation as well as **Fill-in-the-Middle (FIM)** inference — prefix/suffix/middle generation, following [Bavarian et al., 2022 — "Efficient Training of Language Models to Fill in the Middle"](https://arxiv.org/abs/2207.14255) — with a repetition penalty applied during decoding

## Next steps
- Write-ups for each project covering the struggles, key learnings, and takeaways

- Benchmark a wider set of datasets against equivalent PyTorch/TensorFlow implementations
- Reproducing papers like Attention Is All You Need and the Bahdanau attention paper gave a real feel for how these models learn, and an appreciation for how simple the underlying ideas are
- i plan to try and train more models from different aspects of deep learning namely vision and diffusion as well
