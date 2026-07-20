// ─── Embedding Provider Factory ──────────────────────────────────────────────
// Creates the appropriate provider from config, with fallback to local-hash.
//
// Available providers:
//   openai      — text-embedding-3-small (1536d) — $0.02/1M tokens
//   nvidia      — nv-embedqa-e5-v5 (1024d) — FREE tier: 1000 credits/month
//   gemini      — text-embedding-004 (768d) — FREE tier: Google AI Studio
//   local-hash  — FNV-1a sparse vectors (256d) — always free, offline

import type { ContextConfig, EmbeddingProvider } from '../types.ts';
import { LocalHashProvider } from './local-hash.ts';
import { OpenAIProvider } from './openai.ts';
import { NvidiaProvider } from './nvidia.ts';
import { GeminiProvider } from './gemini.ts';

export { LocalHashProvider, normalizeVector, localHashEmbedding } from './local-hash.ts';
export { OpenAIProvider } from './openai.ts';
export { NvidiaProvider } from './nvidia.ts';
export { GeminiProvider } from './gemini.ts';
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
    case 'nvidia': {
      if (!process.env.NVIDIA_API_KEY) {
        console.warn('[codecontext] NVIDIA_API_KEY not set — falling back to local-hash');
        console.warn('[codecontext] Get a FREE key at https://build.nvidia.com');
        return new LocalHashProvider(256);
      }
      return new NvidiaProvider({
        dimensions: config.embedding_dimensions || 1024,
        model: (config.nvidia_model as string) ?? 'nvidia/nv-embedqa-e5-v5',
      });
    }
    case 'gemini': {
      if (!process.env.GOOGLE_API_KEY && !process.env.GEMINI_API_KEY) {
        console.warn('[codecontext] GOOGLE_API_KEY not set — falling back to local-hash');
        console.warn('[codecontext] Get a FREE key at https://aistudio.google.com/apikey');
        return new LocalHashProvider(256);
      }
      return new GeminiProvider({
        dimensions: config.embedding_dimensions || 768,
      });
    }
    case 'local-hash':
      return new LocalHashProvider(config.embedding_dimensions || 256);
    default:
      console.warn(`[codecontext] Unknown provider "${name}" — using local-hash`);
      return new LocalHashProvider(256);
  }
}
