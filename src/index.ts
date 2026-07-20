// ─── CodeContext Engine — Public API ─────────────────────────────────────────
// Intent-aware code retrieval. 50 tokens of perfect context, zero noise.

// Core search
export { search, searchStream, initializeEngine, getEngineHealth } from './searcher.ts';

// Index building
export { buildIndex } from './indexer.ts';

// Config
export { loadConfig, DEFAULT_CONFIG, resolveIndexPath } from './config.ts';

// Formatting
export { formatResults, type OutputMode } from './formatter.ts';

// Evaluation
export { runEvaluation, DEFAULT_EVAL_QUERIES } from './eval.ts';

// Types
export type {
  ContextConfig, SearchResult, SearchOptions, IndexRecord, ScoredCandidate,
  QueryMode, QueryIntent, ProcessedQuery, BuildIndexOptions, BuildIndexResult,
  EvalResult, EvalDetail, EvalExpectation, InvertedIndex, PostingEntry,
  EmbeddingProvider, EmbedOptions, Reranker, QueryPipeline,
} from './types.ts';

// Plugin registration
export { registerEmbeddingProvider, createEmbeddingProvider } from './embeddings/index.ts';
export { registerReranker, createReranker } from './reranker.ts';

// Scoring (for advanced usage / tuning)
export { inferQueryIntent, resultBonus, DEFAULT_RESULT_BONUS_WEIGHTS } from './scoring.ts';
export type { ResultBonusWeights } from './scoring.ts';

// Fusion (for A/B testing)
export { reciprocalRankFusion, fuseWithRRF, weightedSumFusion } from './fusion.ts';
