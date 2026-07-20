// ─── Config ──────────────────────────────────────────────────────────────────
// Configuration loading with env-var overrides.

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import type { ContextConfig } from './types.ts';

export const DEFAULT_CONFIG: ContextConfig = {
  embedding_provider: 'openai',
  embedding_dimensions: 1536,
  openai_model: 'text-embedding-3-small',
  storage_backend: 'json',
  index_path: 'context-index',
  fusion_method: 'rrf',
  rrf_k: 60,
  reranker: 'openai',
  reranker_model: 'gpt-4o-mini',
  reranker_timeout_ms: 3000,
  query_understanding: 'auto',
  query_decomposition: 'auto',
  cache_enabled: true,
  cache_max_entries: 256,
  cache_similarity_threshold: 0.97,
  search_defaultLimit: 10,
  min_chunk_tokens: 50,
  max_chunk_tokens: 800,
  sliding_window_tokens: 400,
  sliding_window_overlap: 100,
};

/** Load config from .context-config.json merged with env-var overrides. */
export function loadConfig(rootDir: string = process.cwd()): ContextConfig {
  const configPath = path.join(rootDir, '.context-config.json');
  let fileConfig: Partial<ContextConfig> = {};

  if (existsSync(configPath)) {
    try {
      fileConfig = JSON.parse(readFileSync(configPath, 'utf8')) as Partial<ContextConfig>;
    } catch {
      // Malformed config — use defaults
    }
  }

  const config: ContextConfig = { ...DEFAULT_CONFIG, ...fileConfig };

  // Env-var overrides (highest priority)
  const env = process.env;
  if (env.CONTEXT_EMBEDDING_PROVIDER) config.embedding_provider = env.CONTEXT_EMBEDDING_PROVIDER;
  if (env.OPENAI_API_KEY && !env.CONTEXT_EMBEDDING_PROVIDER) config.embedding_provider = 'openai';
  if (env.CONTEXT_INDEX_PATH) config.index_path = env.CONTEXT_INDEX_PATH;
  if (env.CONTEXT_FUSION_METHOD) config.fusion_method = env.CONTEXT_FUSION_METHOD as 'rrf' | 'weighted_sum';
  if (env.CONTEXT_RERANKER) config.reranker = env.CONTEXT_RERANKER as 'openai' | 'off';
  if (env.CONTEXT_QUERY_UNDERSTANDING) config.query_understanding = env.CONTEXT_QUERY_UNDERSTANDING as ContextConfig['query_understanding'];
  if (env.CONTEXT_CACHE_ENABLED) config.cache_enabled = env.CONTEXT_CACHE_ENABLED !== '0';
  if (env.CONTEXT_SEARCH_LIMIT) config.search_defaultLimit = Number(env.CONTEXT_SEARCH_LIMIT);

  return config;
}

/** Resolve the active index directory path. */
export function resolveIndexPath(rootDir: string, config: ContextConfig): string {
  return path.join(rootDir, config.index_path);
}
