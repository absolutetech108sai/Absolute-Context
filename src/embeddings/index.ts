// ─── Embedding Provider Factory ──────────────────────────────────────────────
// Creates the appropriate provider from config, with fallback to local-hash.

import type { ContextConfig, EmbeddingProvider } from '../types.ts';
import { LocalHashProvider } from './local-hash.ts';
import { OpenAIProvider } from './openai.ts';

export { LocalHashProvider, normalizeVector, localHashEmbedding } from './local-hash.ts';
export { OpenAIProvider } from './openai.ts';
export type { EmbeddingProvider, EmbedOptions } from './types.ts';

const registry = new Map<string, () => EmbeddingProvider>();

/** Register a custom embedding provider factory. */
export function registerEmbeddingProvider(name: string, factory: () => EmbeddingProvider): void {
  registry.set(name, factory);
}

/** Create an embedding provider from config. Falls back to local-hash on failure. */
export function createEmbeddingProvider(config: ContextConfig): EmbeddingProvider {
  const name = config.embedding_provider;

  // Check custom registry first
  const custom = registry.get(name);
  if (custom) return custom();

  switch (name) {
    case 'openai': {
      if (!process.env.OPENAI_API_KEY) {
        console.warn('[codecontext] OPENAI_API_KEY not set — falling back to local-hash');
        return new LocalHashProvider(256);
      }
      return new OpenAIProvider({
        dimensions: config.embedding_dimensions,
        model: config.openai_model,
      });
    }
    case 'local-hash':
      return new LocalHashProvider(config.embedding_dimensions || 256);
    default:
      console.warn(`[codecontext] Unknown provider "${name}" — using local-hash`);
      return new LocalHashProvider(256);
  }
}
