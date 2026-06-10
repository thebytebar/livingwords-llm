#!/usr/bin/env node

import { program } from 'commander';
import { LivingWordsLLM } from '../core/model.js';
import { configs } from '../core/config.js';

program
  .name('lw-llm')
  .description('God-centered LLM CLI for training and generation')
  .version('0.1.0');

program
  .command('generate')
  .description('Generate text from a prompt')
  .argument('<prompt>', 'Input prompt')
  .option('-m, --max-tokens <number>', 'Max tokens to generate', '100')
  .action(async (prompt, options) => {
    const model = new LivingWordsLLM(configs.pico);
    await model.load(undefined, { silent: true });
    const result = await model.generate(prompt, parseInt(options.maxTokens));
    console.log('\n' + result);
  });

program
  .command('train')
  .description('Train the model on a dataset (character-level transformer)')
  .option('-d, --data <path>', 'Path to training data (.txt)', 'data/bible.txt')
  .option('-e, --epochs <number>', 'Number of full passes over the data', '1')
  .option('--max-iter <number>', 'Maximum training steps (highly recommended to increase)', '800')
  .option('--batch-size <number>', 'Batch size', '16')
  .option('--lr, --learning-rate <number>', 'Learning rate for Adam optimizer', '0.001')
  .option('--eval-interval <number>', 'Steps between printing loss + samples', '100')
  .option('--save-interval <number>', 'Save a checkpoint every N steps (for long runs)', '500')
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

program
  .command('chat')
  .description('Interactive CLI chatbot')
  .option('-l, --load <dir>', 'Directory with saved weights/meta', 'weights')
  .action(async (options) => {
    const { startChat } = await import('./chat.js');
    await startChat(options.load);
  });

program
  .command('serve')
  .description('Start HTTP server with API and web chat UI')
  .option('-p, --port <number>', 'Port to listen on', '3000')
  .option('-l, --load <dir>', 'Directory with saved weights/meta', 'weights')
  .action(async (options) => {
    const { startServer } = await import('./serve.js');
    await startServer(parseInt(options.port), options.load);
  });

program.parse();
