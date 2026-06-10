#!/usr/bin/env node

import { program } from 'commander';
import { LivingWordsLLM } from '../core/model.js';
import { configs } from '../core/config.js';

program
  .name('lw-llm train')
  .description('Train LivingWords LLM on a dataset (character-level GPT)')
  .option('-d, --data <path>', 'Path to training data (.txt)', 'data/bible.txt')
  .option('-e, --epochs <number>', 'Number of full passes over the data', '1')
  .option('--max-iter <number>', 'Total training steps (very useful with small data)', '800')
  .option('--batch-size <number>', 'Batch size', '16')
  .option('--lr, --learning-rate <number>', 'Adam learning rate', '0.001')
  .option('--eval-interval <number>', 'How often to print loss + generated samples', '100')
  .option('--save-interval <number>', 'Save checkpoint every N steps (for long runs)', '500')
  .action(async (options) => {
    const model = new LivingWordsLLM(configs.pico);

    const trainOpts = {
      epochs: parseInt(options.epochs, 10),
      maxIter: parseInt(options.maxIter, 10),
      batchSize: parseInt(options.batchSize, 10),
      learningRate: parseFloat(options.learningRate),
      evalInterval: parseInt(options.evalInterval, 10),
      saveInterval: parseInt(options.saveInterval, 10),
    };

    await model.train(options.data, trainOpts);
    console.log('✅ Training session complete. Model ready for generation.');
  });

program.parse();
