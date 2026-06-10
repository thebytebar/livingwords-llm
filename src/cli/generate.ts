#!/usr/bin/env node

import { program } from 'commander';
import { LivingWordsLLM } from '../core/model.js';
import { configs } from '../core/config.js';

program
  .name('livingwords-llm generate')
  .description('Generate text using LivingWords LLM')
  .argument('<prompt>', 'The prompt to generate from')
  .option('-m, --max-tokens <number>', 'Maximum tokens to generate', '100')
  .action(async (prompt, options) => {
    const model = new LivingWordsLLM(configs.pico);
    const result = await model.generate(prompt, parseInt(options.maxTokens));
    console.log('\n' + result);
  });

program.parse();
