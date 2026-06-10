/**
 * Character-level dataset for LivingWords LLM.
 * Optimized for biblical and theological text.
 */

import * as tf from '@tensorflow/tfjs-node'
import { Dataset, DatasetParams, DatasetGetBatchParams } from './types.js'

export async function createDataset(args: DatasetParams): Promise<Dataset> {
  const { textSource = '', maskZero = true } = args

  const indexShift = maskZero ? 1 : 0

  const text: string = textSource 
  const textSize: number = text.length

  const chars: string[] = Array.from(new Set(text)).sort()
  const vocabSize: number = chars.length

  const stoi = Object.fromEntries(chars.map((ch, i) => [ch, i + indexShift]))
  const itos = Object.fromEntries(chars.map((ch, i) => [i + indexShift, ch]))

  const encode = (s: string) => s.split('').map((c) => stoi[c] || 0)
  const decode = (a: number[]) => a.map((i) => itos[i] || '').join('')

  const data = tf.tensor(encode(text), [textSize], 'int32')
  const dataSize: number = data.shape[0]
  const n = Math.floor(0.9 * textSize)
  const trainData = data.slice(0, n)
  const valData = data.slice(n)

  const getBatch = (args: DatasetGetBatchParams) => tf.tidy(() => {
    const { split, blockSize, batchSize } = args
    const dataSplit = split === 'train' ? trainData : valData
    const maxval = dataSplit.shape[0] - blockSize
    const ix = tf.randomUniform([batchSize], 0, maxval, 'int32')
    const ranges = tf.range(0, blockSize, 1, 'int32').expandDims(0)
    const indices = ix.expandDims(1).add(ranges)
    const x = tf.gather(dataSplit, indices)
    const y = tf.gather(dataSplit, indices.add(tf.scalar(1, 'int32')))
    return { x, y }
  })

  const dispose = () => {
    data.dispose()
    trainData.dispose()
    valData.dispose()
  }

  return {
    vocabSize,
    dataSize,
    vocabulary: chars,
    text,
    getBatch,
    encode,
    decode,
    dispose,
  }
}
