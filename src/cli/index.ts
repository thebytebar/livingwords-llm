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
  .description('Train the model on a dataset')
  .option('-d, --data <path>', 'Path to training data', 'data/bible.txt')
  .option('-e, --epochs <number>', 'Number of epochs', '1')
  .action(async (options) => {
    const model = new LivingWordsLLM(configs.pico);
    await model.train(options.data, parseInt(options.epochs));
    console.log('Training complete!');
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
