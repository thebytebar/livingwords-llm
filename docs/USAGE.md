# LivingWords LLM — Complete Usage Guide

This guide covers everything you need to train the model on new data, use the CLI tools, run the HTTP server/API, and use the web chat interface.

> **Important Context**: LivingWords LLM is a *very small* character-level transformer (~160k parameters in the default "pico" config). It is designed to run on low-power machines. Results are educational and stylistic rather than highly capable. The "God-centered" alignment comes almost entirely from the training data you provide.

## Table of Contents

- [Preparing Training Data](#preparing-training-data)
- [Training the Model](#training-the-model)
- [Training CLI Flags Explained](#training-cli-flags-explained)
- [Understanding Training Progress](#understanding-training-progress)
- [Checkpoints and Model Persistence](#checkpoints-and-model-persistence)
- [Using the CLI](#using-the-cli)
  - [generate](#generate)
  - [chat](#chat)
- [Running the Server & API](#running-the-server--api)
- [The Web Chat Interface](#the-web-chat-interface)
- [Loading Specific Models & Checkpoints](#loading-specific-models--checkpoints)
- [Advanced Usage & Customization](#advanced-usage--customization)
- [Tips for Better Results](#tips-for-better-results)
- [Limitations](#limitations)

---

## Preparing Training Data

The model is **character-level**. Every unique character in your training text becomes part of the vocabulary.

### Requirements

- Plain UTF-8 `.txt` file(s)
- One file is simplest (`--data path/to/corpus.txt`)
- No special formatting required, but consistent line endings and clean text helps

### Recommended Data Strategy (for "God-centered" goals)

1. Start with public domain Scripture (KJV is excellent and freely available).
2. Add sound doctrinal texts: public domain commentaries, creeds (Apostles’, Nicene, Westminster), catechisms, classic sermons (Spurgeon, etc.).
3. Avoid mixing in large amounts of modern casual text if your goal is theological faithfulness.

**Example preparation** (conceptual):

```bash
# You might concatenate multiple sources
cat bible-kjv.txt \
    westminster-confession.txt \
    spurgeon-sermons-selected.txt \
    > data/godly-corpus.txt
```

The current seed file (`data/bible.txt`) is only ~2,650 characters — useful for testing the pipeline but far too small for interesting results.

**Target sizes for meaningful training**:
- Minimum for experimentation: 50k–100k characters
- Good results: 500k+ characters
- Excellent: several million characters of high-quality aligned text

Larger vocabularies (more unique characters, punctuation, numbers) are fine — the model will learn them.

---

## Training the Model

### Basic Command

```bash
npx lw-llm train --data data/bible.txt --epochs 3
```

This uses the default "pico" architecture.

### Recommended Command for Real Training

```bash
npx lw-llm train \
  --data data/godly-corpus.txt \
  --epochs 5 \
  --max-iter 3000 \
  --batch-size 32 \
  --lr 0.0008 \
  --eval-interval 100 \
  --save-interval 250
```

### What Happens During Training

1. The text file is read and a character vocabulary is built (`stoi` / `itos` mappings).
2. A small GPT-style transformer is constructed using the exact vocab size from your data.
3. Training runs next-token prediction (standard language modeling objective).
4. Every `eval-interval` steps you will see:
   - Current loss
   - A generated sample starting from "In the beginning"
5. Every `save-interval` steps a checkpoint is written (see below).
6. At the end the final model + vocabulary is saved to `weights/`.

---

## Training CLI Flags Explained

| Flag                | Default | Description                                                                 | Recommendation |
|---------------------|---------|-----------------------------------------------------------------------------|----------------|
| `--data`            | `data/bible.txt` | Path to your `.txt` training corpus                                        | Use your own larger file |
| `--epochs`          | `1`     | Number of full passes over the data                                         | 3–10 for small data |
| `--max-iter`        | `800`   | **Total optimization steps**. This often matters more than epochs.          | **Increase this** (2000–5000+) |
| `--batch-size`      | `16`    | Number of sequences per gradient step                                       | 16–64 (higher = faster but more memory) |
| `--lr`, `--learning-rate` | `0.001` | Adam optimizer learning rate                                             | 0.0005 – 0.002 |
| `--eval-interval`   | `100`   | How often to print loss + a generation sample                               | 50–200 |
| `--save-interval`   | `500`   | How often to write a checkpoint (see Checkpoints section)                   | 200–500 for long runs |

All flags are available both via the unified `lw-llm` command and the direct `train` script.

---

## Understanding Training Progress

Watch two signals:

1. **Loss** — Should trend downward over time. On this tiny model you might see it go from ~3.5+ down to 1.5–2.0 or lower depending on data size and steps.
2. **Sample quality** — Early samples are gibberish. Later samples should start forming real words, then short phrases, then more coherent biblical/theological language.

Because the model is character-level, it must learn spelling, spacing, and punctuation from scratch. Progress can look slow at first.

---

## Checkpoints and Model Persistence

For long training runs, checkpoints are automatically saved.

### Checkpoint Locations

- Numbered checkpoints: `weights/checkpoint-00250/`, `weights/checkpoint-00500/`, etc.
- Rolling latest: `weights/latest/` (always updated on every checkpoint)
- Final model: `weights/` (plus `weights/latest/` is also updated at the very end)

Each checkpoint directory contains:
- `weights.json` — the neural network weights
- `meta.json` — vocabulary list + model architecture details (so the exact tokenizer can be reconstructed)

### Why Checkpoints Matter

- Protect against crashes during long runs
- Allow you to inspect intermediate model quality
- Provide "latest" as a convenient pointer

---

## Using the CLI

After `npm install livingwords-llm` (or from source after `npm run build`), the binary is `lw-llm`.

### generate

```bash
npx lw-llm generate "In the beginning God created" --max-tokens 120
```

- Automatically loads from `weights/` (or the last trained model).
- `--max-tokens` controls how many new characters to generate (default 100).
- Uses temperature sampling for more interesting output.

### chat

```bash
npx lw-llm chat
# or with a specific checkpoint
npx lw-llm chat --load weights/checkpoint-00125
```

Interactive REPL:
- Type your prompt and press Enter.
- The model generates a continuation.
- Type `exit`, `quit`, `q`, or Ctrl+C to leave.
- Loads from `--load` directory (defaults to `weights/`).

### Other ways to run

From source (development):

```bash
npm run train -- --data data/my.txt --max-iter 500
npm run generate -- "Trust in the Lord"
npm run chat
```

Built version:

```bash
node dist/cli/index.js train ...
```

---

## Running the Server & API

```bash
npx lw-llm serve --port 3000 --load weights
```

Options:
- `--port` / `-p` — HTTP port (default 3000)
- `--load` / `-l` — Directory containing `weights.json` + `meta.json` (default `weights`)

### API Endpoints

| Method | Path            | Description                                      |
|--------|-----------------|--------------------------------------------------|
| POST   | `/api/generate` | Main generation endpoint                         |
| GET    | `/api/health`   | Simple health check                              |
| GET    | `/`             | Serves the web chat UI (see below)               |

#### POST /api/generate

**Request body** (JSON):

```json
{
  "prompt": "The Lord is my shepherd",
  "maxTokens": 140
}
```

**Response**:

```json
{
  "text": "The Lord is my shepherd...",
  "prompt": "The Lord is my shepherd"
}
```

The `text` field contains the full decoded continuation (including the original prompt characters + newly generated ones).

Error responses include an `error` field.

The server loads the model **once** at startup for efficiency.

---

## The Web Chat Interface

When you run `npx lw-llm serve`, a complete single-file web chat UI is served at `http://localhost:3000/`.

### Features of the UI

- Clean, dark, dignified design with gold accents
- Chat bubbles for user and model
- Auto-scrolling message area
- Enter to send, Shift+Enter for new line
- A Proverbs 3:5 verse footer
- Clear disclaimer: "Responses are AI-generated for reflection. Always compare with Scripture."
- Fully client-side (vanilla JavaScript + `fetch`)
- Responsive / mobile friendly
- No external dependencies or build step

The UI simply calls `POST /api/generate` under the hood.

You can open the UI in any modern browser while the server is running.

---

## Loading Specific Models & Checkpoints

Any directory that contains a valid `weights.json` + `meta.json` pair can be loaded.

Examples:

```bash
# CLI
npx lw-llm chat --load weights/checkpoint-00250
npx lw-llm serve --load weights/latest --port 4000

# Programmatic (in your own script)
import { LivingWordsLLM, configs } from 'livingwords-llm';

const model = new LivingWordsLLM(configs.pico);
await model.load('weights/checkpoint-00100');
const reply = await model.generate("What does it mean to walk by faith?", 80);
```

The `meta.json` stores the exact vocabulary and architecture, so the tokenizer will be reconstructed correctly even if you trained on a completely different text corpus.

---

## Advanced Usage & Customization

### Using the library programmatically

```ts
import { LivingWordsLLM, configs } from 'livingwords-llm';

const model = new LivingWordsLLM(configs.pico);

// Train
await model.train('data/my-corpus.txt', {
  epochs: 3,
  maxIter: 2000,
  batchSize: 24,
  learningRate: 8e-4,
  saveInterval: 300,
});

// Or just generate / chat after training
await model.load('weights/');
const text = await model.generate("For God so loved the world", 150);
```

### Switching model size

Currently the CLI hardcodes the `pico` config. You can create your own scripts that use `configs.nano` or pass a custom `ModelConfig`.

### Building from source

```bash
git clone ...
npm install
npm run build
npx lw-llm --help
```

---

## Tips for Better Results

1. **Data is everything** — The model has no external knowledge. Feed it the kind of text you want it to sound like.
2. **Increase `--max-iter`** aggressively for small-to-medium corpora.
3. **Save frequently** (`--save-interval 200` or `250`) during long runs.
4. **Use checkpoints** to evaluate quality at different stages without waiting for full training.
5. **Experiment with learning rate** — too high and loss explodes; too low and progress is glacial.
6. **Clean your data** — remove weird encoding artifacts, excessive repetition, or unrelated text.
7. After training, spend time with `chat` and `generate` to get a feel for the model's "voice".

---

## Limitations

- **Character-level tokenization**: Must learn spelling and common words from scratch. Much less efficient than modern subword tokenizers.
- **Tiny context** (`blockSize=128` in pico): The model only sees the last 128 characters when generating.
- **Very small capacity**: pico has ~160k parameters. It can capture stylistic patterns but will not have deep reasoning or broad knowledge.
- **CPU-only training** (via TensorFlow.js): Training is slower than PyTorch/CUDA setups.
- **No built-in safeguards** or theological guardrails beyond what is present in your training data.
- Generation can repeat or drift after many tokens.

This project is best viewed as an educational, ministry-oriented playground for small on-device language models rather than a production-grade system.

---

## Contributing & Next Steps

Contributions that improve training ergonomics, data pipelines, the web UI, or add thoughtful theological safeguards are especially welcome.

See the main [README.md](../README.md) for the project overview and license (MIT).

---

*May the words of our models — and more importantly, the words we speak — be pleasing in His sight.*