// ─── Searcher ────────────────────────────────────────────────────────────────
// The thin pipeline orchestrator. Connects all modules into the query flow:
// QueryPipeline → Embed → InvertedIndex → BM25 → Vector → RRF → Bonus → Graph → Rerank

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import type {
  ContextConfig, IndexRecord, ScoredCandidate, SearchOptions, SearchResult, StoredIndex,
} from './types.ts';
import { loadConfig, resolveIndexPath } from './config.ts';
import { tokenize } from './tokenizer.ts';
import { createEmbeddingProvider } from './embeddings/index.ts';
import { buildInvertedIndex, precomputeTokenData, retrieveCandidates, type RecordTokenData } from './inverted-index.ts';
import { bm25ScorePrecomputed } from './bm25.ts';
import { scoreVectorCandidates } from './vector-search.ts';
import { fuseWithRRF, weightedSumFusion } from './fusion.ts';
import { inferQueryIntent, resultBonus } from './scoring.ts';
import { applyGraphSidecar, type GraphSidecarMode } from './graph.ts';
import { expandQueryWithConceptMap } from './concept-map.ts';
import { createQueryPipeline } from './query-pipeline.ts';
import { createReranker } from './reranker.ts';
import { SearchCache } from './cache.ts';
import type { InvertedIndex } from './types.ts';

// ─── Engine State (pre-loaded at startup) ────────────────────────────────────

interface EngineState {
  records: IndexRecord[];
  invertedIndex: InvertedIndex;
  tokenData: RecordTokenData[];
  metadata: { document_frequency: Record<string, number>; average_lexical_length: number };
  config: ContextConfig;
  cache: SearchCache;
  builtAt: string;
}

let engineState: EngineState | null = null;

/** Load and pre-process the index. Call once at server startup. */
export function initializeEngine(rootDir: string = process.cwd()): EngineState {
  const config = loadConfig(rootDir);
  const indexPath = resolveIndexPath(rootDir, config);
  const manifestPath = path.join(indexPath, 'manifest.json');
  const recordsPath = path.join(indexPath, 'records.jsonl');

  if (!existsSync(manifestPath) || !existsSync(recordsPath)) {
    throw new Error(`Index not found at ${indexPath}. Run 'node src/cli.ts index' first.`);
  }

  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  const recordsRaw = readFileSync(recordsPath, 'utf8').trim().split('\n');
  const records: IndexRecord[] = recordsRaw.map((line) => JSON.parse(line) as IndexRecord);

  const metadata = {
    document_frequency: (manifest.document_frequency ?? {}) as Record<string, number>,
    average_lexical_length: (manifest.average_lexical_length ?? 1) as number,
  };

  // Pre-build inverted index and token data (one-time cost)
  const invertedIndex = buildInvertedIndex(records);
  const tokenData = precomputeTokenData(records);

  const cache = new SearchCache({
    maxEntries: config.cache_max_entries,
    similarityThreshold: config.cache_similarity_threshold,
  });

  engineState = {
    records,
    invertedIndex,
    tokenData,
    metadata,
    config,
    cache,
    builtAt: (manifest.built_at ?? new Date().toISOString()) as string,
  };

  return engineState;
}

/** Get the current engine state (throws if not initialized). */
function getState(): EngineState {
  if (!engineState) throw new Error('Engine not initialized. Call initializeEngine() first.');
  return engineState;
}

// ─── Main Search Function ────────────────────────────────────────────────────

export async function search(options: SearchOptions): Promise<SearchResult[]> {
  const state = getState();
  const { config, records, invertedIndex, tokenData, metadata, cache } = state;
  const startTime = performance.now();

  const limit = options.limit ?? config.search_defaultLimit;
  const mode = options.mode ?? 'hybrid';
  const ftsOnly = mode === 'local' || options.ftsOnly === true;

  // 1. Check cache
  const optionsKey = `${mode}:${limit}:${options.kindFilter ?? ''}:${options.packageFilter ?? ''}`;
  if (config.cache_enabled) {
    const cached = cache.lookup(options.query, null, optionsKey);
    if (cached) return cached.slice(0, limit);
  }

  // 2. Query pipeline (intent + decomposition)
  const pipeline = createQueryPipeline(config);
  const processed = await pipeline.process(options.query, config);
  const intent = processed.intent;

  // 3. Concept expansion
  const rootDir = options.rootDir ?? process.cwd();
  const expansion = expandQueryWithConceptMap(options.query, rootDir);
  const effectiveQuery = expansion.applied ? expansion.expandedQuery : options.query;

  // 4. Embed query
  let queryVector: number[] | null = null;
  if (!ftsOnly) {
    try {
      const provider = createEmbeddingProvider(config);
      const [embedded] = await provider.embed([effectiveQuery], { inputType: 'query' });
      queryVector = embedded ?? null;
    } catch {
      queryVector = null; // Degrade to BM25-only
    }
  }

  // 5. Retrieve candidates via inverted index
  const queryTokens = tokenize(effectiveQuery);
  const candidateIds = retrieveCandidates(invertedIndex, queryTokens, 500, ftsOnly ? 0 : 200);

  // 6. Score candidates with BM25
  const bm25Scored: ScoredCandidate[] = candidateIds.map((docId) => {
    const record = records[docId]!;
    const td = tokenData[docId]!;
    const lexical = bm25ScorePrecomputed(
      queryTokens, td.tokenCounts, td.tokenTotal,
      metadata.document_frequency, metadata.average_lexical_length, invertedIndex.docCount,
    );
    return { ...record, docId, _lexical: lexical, _semantic: 0, _score: 0 };
  });

  // 7. Score candidates with vector similarity
  let vectorScored: ScoredCandidate[] = [];
  if (queryVector && !ftsOnly) {
    const vResults = scoreVectorCandidates(queryVector, records, candidateIds);
    vectorScored = vResults.map((v) => ({
      ...records[v.docId]!,
      docId: v.docId,
      _lexical: 0,
      _semantic: v.score,
      _score: 0,
    }));
  }

  // 8. Fusion: RRF or weighted sum
  let fused: ScoredCandidate[];
  if (ftsOnly) {
    fused = bm25Scored.map((c) => ({ ...c, _score: c._lexical }));
    fused.sort((a, b) => b._score - a._score);
  } else if (config.fusion_method === 'rrf' && vectorScored.length > 0) {
    const bm25Sorted = [...bm25Scored].sort((a, b) => b._lexical - a._lexical);
    const vectorSorted = [...vectorScored].sort((a, b) => b._semantic - a._semantic);
    const allCandidates = mergeById(bm25Scored, vectorScored);
    fused = fuseWithRRF(bm25Sorted, vectorSorted, allCandidates, config.rrf_k);
  } else {
    // Weighted sum fallback
    const allCandidates = mergeById(bm25Scored, vectorScored);
    fused = weightedSumFusion(allCandidates, 0.55, 0.45, ftsOnly);
  }

  // 9. Apply intent-aware bonus scoring
  fused = fused.map((candidate) => {
    const td = tokenData[candidate.docId];
    const bonus = resultBonus(effectiveQuery, candidate, intent, td);
    return { ...candidate, _score: candidate._score + bonus };
  });
  fused.sort((a, b) => b._score - a._score);

  // 10. Filter by kind/package
  if (options.kindFilter) {
    fused = fused.filter((c) => c.kind === options.kindFilter);
  }
  if (options.packageFilter) {
    fused = fused.filter((c) => c.package_name === options.packageFilter);
  }

  // 11. Rerank (optional, GPT-4o-mini)
  if (options.rerank && config.reranker !== 'off') {
    const reranker = createReranker(config);
    fused = await reranker.rerank(options.query, fused, limit);
  }

  // 11b. Collapse multiple chunks from the same file — keep the top-scoring
  // chunk per file so results show distinct files, not the same file 3×.
  const seenFiles = new Set<string>();
  fused = fused.filter((c) => {
    if (seenFiles.has(c.file_path)) return false;
    seenFiles.add(c.file_path);
    return true;
  });

  // 12. Format results
  const results: SearchResult[] = fused.slice(0, limit).map((c) => ({
    file_path: c.file_path,
    line_start: c.line_start,
    line_end: c.line_end,
    kind: c.kind,
    package_name: c.package_name,
    symbols: c.symbols,
    heading: c.heading,
    text: c.text,
    token_estimate: c.token_estimate,
    _score: c._score,
    _distance: 1 - (c._semantic_normalized ?? c._semantic ?? 0),
    _backend: 'json' as const,
    citation: `${c.file_path}:${c.line_start}-${c.line_end}`,
  }));

  // 13. Store in cache
  if (config.cache_enabled) {
    cache.store(options.query, queryVector, optionsKey, results);
  }

  return results;
}

/** Streaming search — yields results as they're scored. */
export async function* searchStream(options: SearchOptions): AsyncGenerator<SearchResult> {
  const results = await search(options);
  for (const result of results) {
    yield result;
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function mergeById(bm25: ScoredCandidate[], vector: ScoredCandidate[]): ScoredCandidate[] {
  const map = new Map<number, ScoredCandidate>();
  for (const c of bm25) {
    map.set(c.docId, c);
  }
  for (const c of vector) {
    const existing = map.get(c.docId);
    if (existing) {
      map.set(c.docId, { ...existing, _semantic: c._semantic });
    } else {
      map.set(c.docId, c);
    }
  }
  return [...map.values()];
}

/** Get engine health info. */
export function getEngineHealth(): Record<string, unknown> {
  if (!engineState) return { status: 'not_initialized' };
  return {
    status: 'ready',
    recordCount: engineState.records.length,
    builtAt: engineState.builtAt,
    embeddingProvider: engineState.config.embedding_provider,
    fusionMethod: engineState.config.fusion_method,
    cache: engineState.cache.stats(),
  };
}
