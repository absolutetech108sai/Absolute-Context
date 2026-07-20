// ─── Local Hash Embedding Provider ──────────────────────────────────────────
// Zero-infra FNV-1a hash-based sparse vectors. Works offline, no API key needed.

import type { EmbeddingProvider, EmbedOptions } from '../types.ts';
import { tokenize, createTokenCountMap } from '../tokenizer.ts';

function fnv1a(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  if (!magnitude) return vector;
  return vector.map((v) => v / magnitude);
}

export function localHashEmbedding(text: string, dimensions: number): number[] {
  const vector = new Array<number>(dimensions).fill(0);
  const tokenCounts = createTokenCountMap(tokenize(text));
  for (const [token, count] of tokenCounts.entries()) {
    const hash = fnv1a(token);
    const slot = hash % dimensions;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vector[slot] += sign * (1 + Math.log1p(count));
  }
  return normalizeVector(vector);
}

export class LocalHashProvider implements EmbeddingProvider {
  readonly name = 'local-hash';
  readonly dimensions: number;

  constructor(dimensions: number = 256) {
    this.dimensions = dimensions;
  }

  async embed(texts: string[], _opts: EmbedOptions): Promise<number[][]> {
    return texts.map((text) => localHashEmbedding(text, this.dimensions));
  }
}
