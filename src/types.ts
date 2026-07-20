// ─── CodeContext Engine Types ────────────────────────────────────────────────
// All shared interfaces for the retrieval engine pipeline.

// ─── Index Types ─────────────────────────────────────────────────────────────

export type IndexKind =
  | 'agent_rules'
  | 'code'
  | 'conversation'
  | 'doc'
  | 'graph'
  | 'schema'
  | 'test'
  | 'tooling';

export interface IndexRecord {
  text: string;
  vector: number[];
  file_path: string;
  chunk_id: string;
  kind: IndexKind;
  package_name: string;
  symbols: string;
  heading: string;
  community_id: number;
  line_start: number;
  line_end: number;
  last_modified: string | null;
  recency_bucket: number;
  token_estimate: number;
  search_text: string;
  concept_text: string;
}

export interface SearchResult {
  file_path: string;
  line_start: number;
  line_end: number;
  kind: IndexKind;
  package_name: string;
  symbols: string;
  heading: string;
  text: string;
  token_estimate: number;
  _score: number;
  _distance: number;
  _backend: 'json' | 'lancedb';
  citation?: string;
}

export interface ScoredCandidate extends IndexRecord {
  _lexical: number;
  _semantic: number;
  _lexical_normalized?: number;
  _semantic_normalized?: number;
  _score: number;
  _distance?: number;
  _rrf_score?: number;
  _rerank_score?: number;
  _graph_sidecar_boost?: number;
  _graph_sidecar_injected?: boolean;
  docId: number;
}

// ─── Config ──────────────────────────────────────────────────────────────────

export interface ContextConfig {
  embedding_provider: string;
  embedding_dimensions: number;
  openai_model: string;
  storage_backend: string;
  index_path: string;
  fusion_method: 'rrf' | 'weighted_sum';
  rrf_k: number;
  reranker: 'openai' | 'off';
  reranker_model: string;
  reranker_timeout_ms: number;
  query_understanding: 'auto' | 'llm' | 'regex' | 'off';
  query_decomposition: 'auto' | 'always' | 'off';
  cache_enabled: boolean;
  cache_max_entries: number;
  cache_similarity_threshold: number;
  search_defaultLimit: number;
  min_chunk_tokens: number;
  max_chunk_tokens: number;
  sliding_window_tokens: number;
  sliding_window_overlap: number;
  nvidia_model?: string;
  nvidia_batch_size?: number;
  ollama_host?: string;
  ollama_model?: string;
  ollama_batch_size?: number;
  ollama_max_retries?: number;
  [key: string]: unknown;
}

// ─── Query Types ─────────────────────────────────────────────────────────────

export type QueryIntent =
  | 'documentation'
  | 'data_model'
  | 'runtime_semantic'
  | 'implementation'
  | 'commit_message';

export type QueryMode = 'local' | 'global' | 'hybrid';

export interface SearchOptions {
  rootDir?: string;
  query: string;
  mode?: QueryMode;
  limit?: number;
  kindFilter?: string;
  packageFilter?: string;
  ftsOnly?: boolean;
  rerank?: boolean;
}

export interface ProcessedQuery {
  original: string;
  subQueries: string[];
  intent: QueryIntent;
  expandedTerms: string[];
  embedding?: number[];
}

// ─── Plugin Interfaces ───────────────────────────────────────────────────────

export interface EmbedOptions {
  inputType: 'passage' | 'query';
  model?: string;
}

export interface EmbeddingProvider {
  readonly name: string;
  readonly dimensions: number;
  embed(texts: string[], opts: EmbedOptions): Promise<number[][]>;
}

export interface Reranker {
  readonly name: string;
  rerank(query: string, candidates: ScoredCandidate[], limit: number): Promise<ScoredCandidate[]>;
}

export interface QueryPipeline {
  process(query: string, config: ContextConfig): Promise<ProcessedQuery>;
}

// ─── Index Build Types ───────────────────────────────────────────────────────

export interface BuildIndexOptions {
  rootDir?: string;
  incremental?: boolean;
  /** Only index under these top-level directories (e.g. ['apps','packages','scripts']).
   *  Omit to index everything. Focuses a code engine on actual source. */
  includeDirs?: string[];
  /** Extra directory names to skip (merged with built-in ignores). */
  excludeDirs?: string[];
  /** Safety valve: stop after this many records. */
  maxRecords?: number;
}

export interface BuildIndexResult {
  indexFile: string;
  recordCount: number;
  countsByKind: Record<string, number>;
  warnings: string[];
  durationMs: number;
}

export interface IndexMetadata {
  average_lexical_length: number;
  document_frequency: Record<string, number>;
}

export interface StoredIndex {
  version: number;
  backend: string;
  built_at: string;
  git_sha: string;
  metadata: IndexMetadata;
  records: IndexRecord[];
  record_count: number;
  embedding_provider: string;
  embedding_model: string;
  embedding_dimensions: number;
}

// ─── Eval Types ──────────────────────────────────────────────────────────────

export interface EvalExpectation {
  query: string;
  expected_file: string;
}

export interface EvalDetail {
  query: string;
  expected_file: string;
  rank: number | null;
  hit: boolean;
}

export interface EvalResult {
  total: number;
  precisionAt5: number;
  mrr: number;
  ndcgAt5: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  details: EvalDetail[];
}

// ─── Inverted Index Types ────────────────────────────────────────────────────

export interface PostingEntry {
  docId: number;
  tf: number;
}

export interface InvertedIndex {
  postings: Map<string, PostingEntry[]>;
  idf: Map<string, number>;
  docLengths: number[];
  avgDocLength: number;
  docCount: number;
}

// ─── Cache Types ─────────────────────────────────────────────────────────────

export interface CacheEntry {
  results: SearchResult[];
  createdAtMs: number;
  vector?: number[];
}
