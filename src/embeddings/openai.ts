// ─── OpenAI Embedding Provider ───────────────────────────────────────────────
// Uses text-embedding-3-small (1536d) via the OpenAI API. Hackathon primary.

import type { EmbeddingProvider, EmbedOptions } from '../types.ts';
import { normalizeVector } from './local-hash.ts';

export class OpenAIProvider implements EmbeddingProvider {
  readonly name = 'openai';
  readonly dimensions: number;
  private model: string;
  private apiKey: string;

  constructor(opts: { dimensions?: number; model?: string; apiKey?: string } = {}) {
    this.dimensions = opts.dimensions ?? 1536;
    this.model = opts.model ?? 'text-embedding-3-small';
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? '';
  }

  async embed(texts: string[], opts: EmbedOptions): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error('OPENAI_API_KEY is not set');
    }

    const model = opts.model ?? this.model;
    const batchSize = 100; // OpenAI max batch size
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model,
          input: batch,
          dimensions: this.dimensions,
        }),
      });

      if (!response.ok) {
        const body = await response.text().catch(() => '');
        throw new Error(`OpenAI embeddings failed (${response.status}): ${body}`);
      }

      const payload = (await response.json()) as {
        data?: Array<{ embedding: number[] }>;
      };

      if (!Array.isArray(payload.data)) {
        throw new Error('OpenAI response did not include data array');
      }

      allEmbeddings.push(...payload.data.map((item) => normalizeVector(item.embedding)));
    }

    if (allEmbeddings.length !== texts.length) {
      throw new Error(`OpenAI returned ${allEmbeddings.length} embeddings for ${texts.length} texts`);
    }

    return allEmbeddings;
  }
}
