import * as tf from '@tensorflow/tfjs-node';

export interface ModelConfig {
  nLayer: number;
  nHead: number;
  nEmbd: number;
  vocabSize: number;
  blockSize: number;
  embdDropout?: number;
  residDropout?: number;
  attnDropout?: number;
}

export interface LivingWordsModel {
  params: ModelConfig;
  apply: (x: tf.Tensor) => tf.Tensor;
  generate: (args: any, onGenerateChar?: (token: number) => void) => Promise<tf.Tensor>;
  loss: (x: tf.Tensor, y: tf.Tensor) => tf.Tensor;
  build: () => void;
  summary: () => { params: number };
  // Add more helpers as needed
}
