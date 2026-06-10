# Educational Guide: Core Concepts in LivingWords LLM

This document explains the fundamental ideas behind how this LLM works. It is written for developers who want to understand the "why" and "how" of building a small language model from scratch.

LivingWords LLM is a **tiny decoder-only transformer** implemented in TypeScript using TensorFlow.js. It is heavily inspired by Andrej Karpathy’s nanoGPT and the "homemade-gpt-js" style implementations. The entire model is designed to be understandable and runnable on ordinary laptops.

---

## Table of Contents

1. [What is a Language Model?](#what-is-a-language-model)
2. [Tokenization: Turning Text into Numbers](#tokenization-turning-text-into-numbers)
3. [Embeddings: Giving Tokens Meaning and Position](#embeddings-giving-tokens-meaning-and-position)
4. [The Transformer Architecture](#the-transformer-architecture)
5. [Self-Attention: The Heart of the Model](#self-attention-the-heart-of-the-model)
6. [The Full Transformer Block](#the-full-transformer-block)
7. [Training: Next-Token Prediction](#training-next-token-prediction)
8. [Loss Function: Measuring How "Surprised" the Model Is](#loss-function-measuring-how-surprised-the-model-is)
9. [Generation: Making the Model Talk](#generation-making-the-model-talk)
10. [Why So Small? (Pico vs Nano Configs)](#why-so-small-pico-vs-nano-configs)
11. [Putting It All Together](#putting-it-all-together)
12. [Further Reading](#further-reading)

---

## What is a Language Model?

At its core, a language model is a system that learns to predict the **next piece of text** given some previous text.

You give it:

> "In the beginning God created the"

It tries to predict what comes next (e.g., "heavens").

Modern LLMs do this at massive scale. LivingWords does the exact same thing, just at a tiny educational scale using only characters as the basic unit.

The magic comes from turning this simple idea into a neural network that can capture patterns, grammar, style, and (in this case) theological tone — all learned purely from the training data.

---

## Tokenization: Turning Text into Numbers

Computers don't understand letters. They need numbers.

### Character-Level Tokenization

This project uses the simplest possible approach: **character-level tokenization**.

See [src/core/dataset.ts](../src/core/dataset.ts):

```ts
const chars = Array.from(new Set(text)).sort();   // unique characters
const stoi = ...;  // character → integer ID
const itos = ...;  // integer ID → character

const encode = (s: string) => s.split('').map(c => stoi[c]);
const decode = (ids: number[]) => ids.map(i => itos[i]).join('');
```

Example with the tiny seed data:
- `"God"` → `[7, 24, 3]` (or whatever IDs are assigned)
- The model never sees the letter "G" directly — it only sees the number `7`.

**Why character level?**
- Extremely simple to implement.
- No external tokenizer libraries needed.
- Great for learning — you can literally watch the model learn to spell.
- **Downsides**: The model has to learn that "the" is a word, that "God" is important, etc., one character at a time. Real production models (GPT-4, Claude, etc.) use **subword tokenization** (BPE, SentencePiece) which is much more efficient.

The vocabulary size in this project is dynamic — it is exactly the number of unique characters in your training file (usually 40–80 for English theological text).

---

## Embeddings: Giving Tokens Meaning and Position

Once we have token IDs, we turn them into **vectors** (lists of numbers) that the neural network can work with.

There are two kinds of embeddings in this model:

### 1. Token Embeddings (`wte`)

Each possible character ID gets its own learned vector.

```ts
// In gpt-model.ts
wte: tf.layers.embedding({ inputDim: vocabSize + 1, outputDim: nEmbd, ... })
```

This is like a lookup table: ID `7` → a 64-dimensional vector (in the pico config).

### 2. Positional Embeddings (`wpe`)

Transformers have **no built-in sense of order**. They process all tokens in parallel.

To fix this, we add a second embedding that tells the model "this is position 5 in the sequence".

```ts
wpe: tf.layers.embedding({ inputDim: blockSize, outputDim: nEmbd, ... })
```

The final input to the transformer is:

```
input = token_embedding + positional_embedding
```

This combination lets the model know both *what* the token is and *where* it appears.

---

## The Transformer Architecture

The model used here is a **decoder-only transformer** (the same family as GPT-2, GPT-3, Llama, etc.).

Key properties:
- It is **causal** — it can only look at past and present tokens, never future ones (this is what allows autoregressive generation).
- It is built by stacking identical **Transformer Blocks**.

The high-level flow (see `apply` in [src/core/gpt-model.ts](../src/core/gpt-model.ts)):

1. Token + Positional embeddings
2. Dropout
3. Stack of `nLayer` Transformer Blocks
4. Final Layer Normalization
5. Language Model Head (linear layer) → logits over the vocabulary

---

## Self-Attention: The Heart of the Model

This is the most important innovation in modern LLMs.

### Intuition

Self-attention lets every token "look at" every other token (in the allowed range) and decide how much it should pay attention to them.

When predicting the next word after "In the beginning God created the", the model might want to pay special attention to the word "God" even if it is several tokens back.

### How It Works (Simplified)

Inside `CausalSelfAttention` ([src/core/gpt-model.ts](../src/core/gpt-model.ts) lines ~150-186):

1. **Linear projections** turn the input into three things for each token:
   - **Query (Q)**: "What am I looking for?"
   - **Key (K)**: "What do I contain?"
   - **Value (V)**: "What information do I actually provide?"

2. **Attention scores** are computed as:
   ```
   scores = (Q @ K^T) / sqrt(head_size)
   ```

3. **Causal Mask** — this is crucial:
   ```ts
   att = tf.where(bias...equal(0), -Infinity, att)
   att = tf.softmax(att)
   ```
   Future positions are set to `-∞` so they become 0 after softmax. The model literally cannot "cheat" by looking ahead.

4. **Weighted sum** of the Values using the attention weights.

### Multi-Head Attention

Instead of doing this once, we do it in parallel with multiple "heads" (4 heads in pico config). Each head can learn to focus on different kinds of relationships (syntax, long-range references, theological themes, etc.).

After the heads run, their outputs are concatenated and projected back down.

---

## The Full Transformer Block

A single block (see `Block` function) contains:

```
x = LayerNorm(x)
x = x + Attention(x)          // residual connection

x = LayerNorm(x)
x = x + FeedForward(x)        // residual connection
```

### Why Residual Connections?

They allow gradients to flow easily through deep networks (the "skip connection" idea from ResNet).

### Why Layer Normalization?

It stabilizes training by keeping activations from exploding or vanishing.

### Feed-Forward Network

After attention mixes information across tokens, the feed-forward layer processes each token **independently** with a small neural network (usually expanding to 4× the embedding size with a GELU activation, then projecting back).

---

## Training: Next-Token Prediction

Training is surprisingly simple in concept.

We take the training text and create many examples of the form:

```
x = "In the beginning God created the"     →    y = "n the beginning God created the "
x = "In the beginning God created the h"   →    y = " the beginning God created the he"
```

See `getBatch` in [src/core/dataset.ts](../src/core/dataset.ts). It creates these shifted pairs efficiently using tensor operations.

The model is trained to output a probability distribution over every possible next character for every position.

---

## Loss Function: Measuring How "Surprised" the Model Is

We use **cross-entropy loss** (see `loss` method in gpt-model.ts).

For each position, the model outputs raw scores called **logits**. We turn these into probabilities with softmax, then measure how much probability mass the model assigned to the *correct* next character.

Low loss = the model is usually putting high probability on the right next character.

The optimizer (Adam) then slightly adjusts every weight in the network to reduce this loss.

---

## Generation: Making the Model Talk

Once trained, we use the model **autoregressively**:

1. Encode the prompt into token IDs.
2. Feed the current sequence into the model.
3. Look only at the logits for the *last* position.
4. Convert to probabilities.
5. Sample a token (or pick the most likely one).
6. Append it to the sequence.
7. Repeat.

See the `generate` method in [src/core/gpt-model.ts](../src/core/gpt-model.ts).

### Sampling Tricks Used Here

- **Temperature**: Divides the logits before softmax. Lower temperature = more confident (less random). Higher = more creative/chaotic.
- **doSample**: Whether to sample from the distribution or always take the argmax.
- The code also supports `topK` (though not heavily used in the current CLI).

Because the model is character-level, generation can produce spelling mistakes or strange word breaks — this is normal at this scale.

---

## Why So Small? (Pico vs Nano Configs)

See [src/core/config.ts](../src/core/config.ts):

| Config | Layers | Embedding Dim | Heads | Block Size | Approx. Params |
|--------|--------|---------------|-------|------------|----------------|
| pico   | 3      | 64            | 4     | 128        | ~160k          |
| nano   | 6      | 128           | 6     | 256        | ~1M+           |

These are deliberately tiny compared to real models (GPT-2 small was 124 million parameters).

**Benefits of tiny models**:
- Train on a laptop CPU in minutes to hours.
- Easy to understand every weight.
- Can run inference with almost no resources.
- Perfect for learning and for very narrow domains (like biblical language patterns).

The tradeoff is capability. This model will never write a full sermon or reason deeply — but it *can* learn the "flavor" of Scripture remarkably well with enough data and training.

---

## Putting It All Together

The full data flow when you run `npx lw-llm generate "Trust in the"`:

1. `LivingWordsLLM.generate()` ([src/core/model.ts](../src/core/model.ts)) calls `initTokenizer()` which builds `encode`/`decode` from the saved vocabulary.
2. Prompt is encoded to token IDs.
3. The IDs are wrapped in a tensor and passed to the raw GPT model's `generate()` method.
4. Inside the model: embeddings → stack of transformer blocks (attention + FFN) → final head → logits.
5. Sampling loop appends new tokens one by one.
6. Final token IDs are decoded back into text using the same vocabulary.

The training path is similar but uses the `trainLivingWordsLLM` function in [src/core/trainer.ts](../src/core/trainer.ts) and the dataset's `getBatch` to create (x, y) pairs.

---

## Further Reading

- Andrej Karpathy’s **nanoGPT** (the spiritual parent of this code): https://github.com/karpathy/nanoGPT
- "Let's build GPT: from scratch, in code, spelled out" (YouTube) — highly recommended
- "Attention Is All You Need" (the original Transformer paper)
- "The Illustrated Transformer" by Jay Alammar (excellent visual explanations)
- Karpathy’s "Neural Networks: Zero to Hero" series

---

This tiny implementation deliberately exposes every moving part. By reading the code in `gpt-model.ts`, `dataset.ts`, and `trainer.ts` alongside this document, you can develop a genuine understanding of how real (much larger) language models work under the hood.

If something is still unclear, the best next step is usually to add `console.log` statements or reduce the model to a single layer and watch what happens during a short training run.

May your study of these systems lead to deeper appreciation of both the technology and the One who gave us language in the first place.