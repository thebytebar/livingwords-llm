# LivingWords LLM

**Open-source God-centered LLM** aligned with Christian theological doctrine.

A lightweight transformer model designed to run on low-power machines. The project includes the model itself, training on new documents, a CLI chatbot, a server/API for serving the model, and a web-based chat interface.

## Features
- Train small transformer models locally in Node.js (char-level, CPU-friendly)
- CLI chatbot (`lw-llm chat`) for interacting with the bundled LLM
- Server/API mode (`lw-llm serve`) — REST API + served web chat UI
- Web-based chat interface (single-file, works in any browser)
- Full training pipeline for new documents (with automatic weights + vocab save/load)
- Optimized for machines without a lot of power (tiny "pico" config ~160k params)

## Quick Start

```bash
npm install livingwords-llm
```

### CLI Usage
```bash
# Train on custom data (saves weights + vocab to ./weights)
npx lw-llm train --data data/bible.txt --epochs 3

# Generate text (auto-loads saved weights if present)
npx lw-llm generate "In the beginning God created"

# Interactive chatbot
npx lw-llm chat

# Start server + web chat UI (http://localhost:3000)
npx lw-llm serve --port 3000
```

The web UI is served at `/` and the generation API at `POST /api/generate` (JSON: `{ "prompt": "...", "maxTokens": 120 }`).

## Project Structure
- `src/core/` — model (GPT transformer), trainer, dataset (char-level), config, persistence
- `src/cli/` — `index.ts` (unified), `train`, `generate`, `chat`, `serve`
- `data/bible.txt` — seed training corpus (expand for better results)
- `weights/` — auto-created; contains `weights.json` + `meta.json` (vocab) after training (git-ignored)

> Tip: For higher quality faithful output, train for more epochs on a larger corpus of Scripture + sound doctrine. The included pico model is intentionally tiny for low-power machines.
See src/ for the core implementation.

## Contributing
We welcome contributions! Fork the repo, make PRs focused on theological accuracy, performance, and usability for ministry.

## License
MIT


