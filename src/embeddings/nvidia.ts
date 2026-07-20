// ─── NVIDIA NIM Embedding Provider ──────────────────────────────────────────
// Free tier: 1000 credits/month on build.nvidia.com
// Model: nvidia/nv-embedqa-e5-v5 (1024d)

import type { EmbeddingProvider, EmbedOptions } from '../types.ts';
import { normalizeVector } from './local-hash.ts';

export class NvidiaProvider implements EmbeddingProvider {
  readonly name = 'nvidia';
  readonly dimensions: number;
  private model: string;
  private apiKey: string;
  private batchSize: number;

  constructor(opts: { dimensions?: number; model?: string; apiKey?: string; batchSize?: number } = {}) {
    this.dimensions = opts.dimensions ?? 1024;
    this.model = opts.model ?? 'nvidia/nv-embedqa-e5-v5';
    this.apiKey = opts.apiKey ?? process.env.NVIDIA_API_KEY ?? '';
    this.batchSize = opts.batchSize ?? 50;
  }

  async embed(texts: string[], opts: EmbedOptions): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error('NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com');
    }

    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      let lastError: unknown = null;

      // Retry up to 2 times
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const response = await fetch('https://integrate.api.nvidia.com/v1/embeddings', {
            method: 'POST',
            headers: {
              'content-type': 'application/json',
              authorization: `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify({
              model: this.model,
              input: batch,
              input_type: opts.inputType === 'query' ? 'query' : 'passage',
              encoding_format: 'float',
            }),
          });

          if (!response.ok) {
            throw new Error(`NVIDIA NIM failed (${response.status})`);
          }

          const payload = (await response.json()) as {
            data?: Array<{ embedding: number[] }>;
          };

          if (!Array.isArray(payload.data)) {
            throw new Error('NVIDIA response missing data array');
          }

          allEmbeddings.push(...payload.data.map((item) => normalizeVector(item.embedding)));
          lastError = null;
          break;
        } catch (err) {
          lastError = err;
          if (attempt < 2) {
            await new Promise((r) => setTimeout(r, 300 * (attempt + 1)));
          }
        }
      }

      if (lastError) throw lastError;
    }

    return allEmbeddings;
  }
}
