import * as tf from '@tensorflow/tfjs-node';
import { ModelConfig } from './config.js';
import { GPT } from './gpt-model.js'; // Will contain the full transformer

export class LivingWordsLLM {
  private config: ModelConfig;
  private model: any; // GPT model instance
  private isBuilt: boolean = false;

  constructor(config: ModelConfig) {
    this.config = config;
    console.log('🌟 LivingWordsLLM initialized with God-centered config:', config);
  }

  private async build() {
    if (this.isBuilt) return;
    // Initialize the full GPT transformer
    this.model = GPT({
      nLayer: this.config.nLayer,
      nHead: this.config.nHead,
      nEmbd: this.config.nEmbd,
      vocabSize: this.config.vocabSize,
      blockSize: this.config.blockSize,
      embdDropout: this.config.dropout,
      residDropout: this.config.dropout,
      attnDropout: this.config.dropout,
    });
    this.model.build();
    this.isBuilt = true;
    console.log('✅ Model built. Parameter count:', this.model.summary().params);
  }

  async train(dataPath: string, epochs: number = 1): Promise<void> {
    await this.build();
    console.log(`🙏 Training LivingWordsLLM on ${dataPath} for ${epochs} epochs...`);

    // Full training loop integration
    const { trainLivingWordsLLM } = await import('./trainer.js');
    const trainedModel = await trainLivingWordsLLM(this.config, dataPath, { epochs, maxIter: 500 });
    
    // Update internal model reference
    this.model = trainedModel;
    console.log('📖 Training aligned with biblical doctrine. Ready for faithful generation.');
  }

  async generate(prompt: string, maxTokens: number = 100): Promise<string> {
    await this.build();
    console.log(`🤖 Generating God-centered response for: "${prompt}"`);

    // TODO: Full generation with tokenizer
    // Placeholder for now
    return prompt + "\n\n[LivingWords continuation: A faithful, scripture-aligned response will be generated here...]";
  }
}
