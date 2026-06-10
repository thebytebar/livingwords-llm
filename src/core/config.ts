export interface ModelConfig {
  vocabSize: number;
  nEmbd: number; // embedding dimension (was nEmb)
  nHead: number;
  nLayer: number;
  blockSize: number; // context length
  dropout?: number;
}

export const configs = {
  pico: {
    vocabSize: 256, // char-level
    nEmbd: 64,
    nHead: 4,
    nLayer: 3,
    blockSize: 128,
    dropout: 0.1,
  } as ModelConfig,
  nano: {
    vocabSize: 256,
    nEmbd: 128,
    nHead: 6,
    nLayer: 6,
    blockSize: 256,
    dropout: 0.1,
  } as ModelConfig,
  // Add more as needed
};
