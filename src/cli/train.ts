#!/usr/bin/env node

import { program } from 'commander';
import { LivingWordsLLM } from '../core/model.js';
import { configs } from '../core/config.js';

program
  .name('lw-llm train')
  .description('Train LivingWords LLM on dataset')
  .option('-d, --data <path>', 'Path to training data (.txt)', 'data/bible.txt')
  .option('-e, --epochs <number>', 'Number of epochs', '1')
  .action(async (options) => {
    const model = new LivingWordsLLM(configs.pico);
    await model.train(options.data, parseInt(options.epochs));
    console.log('✅ Training session complete. Model ready for generation.');
  });

program.parse();
