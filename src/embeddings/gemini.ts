// ─── Google Gemini Embedding Provider ────────────────────────────────────────
// Free tier: Google AI Studio (aistudio.google.com) — generous rate limits.
// Model: text-embedding-004 (768d)

import type { EmbeddingProvider, EmbedOptions } from '../types.ts';
import { normalizeVector } from './local-hash.ts';

export class GeminiProvider implements EmbeddingProvider {
  readonly name = 'gemini';
  readonly dimensions: number;
  private model: string;
  private apiKey: string;

  constructor(opts: { dimensions?: number; model?: string; apiKey?: string } = {}) {
    this.dimensions = opts.dimensions ?? 768;
    this.model = opts.model ?? 'text-embedding-004';
    this.apiKey = opts.apiKey ?? process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY ?? '';
  }

  async embed(texts: string[], opts: EmbedOptions): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error('GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey');
    }

    const allEmbeddings: number[][] = [];
    const batchSize = 100; // Gemini supports up to 100 texts per batch

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);

      const body = {
        requests: batch.map((text) => ({
          model: `models/${this.model}`,
          content: { parts: [{ text }] },
          taskType: opts.inputType === 'query' ? 'RETRIEVAL_QUERY' : 'RETRIEVAL_DOCUMENT',
        })),
      };

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:batchEmbedContents?key=${this.apiKey}`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        },
      );

      if (!response.ok) {
        const errText = await response.text().catch(() => '');
        throw new Error(`Gemini embeddings failed (${response.status}): ${errText}`);
      }

      const payload = (await response.json()) as {
        embeddings?: Array<{ values: number[] }>;
      };

      if (!Array.isArray(payload.embeddings)) {
        throw new Error('Gemini response missing embeddings array');
      }

      allEmbeddings.push(...payload.embeddings.map((e) => normalizeVector(e.values)));
    }

    if (allEmbeddings.length !== texts.length) {
      throw new Error(`Gemini returned ${allEmbeddings.length} embeddings for ${texts.length} texts`);
    }

    return allEmbeddings;
  }
}
