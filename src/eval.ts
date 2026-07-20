// ─── Evaluation ──────────────────────────────────────────────────────────────
// Built-in quality gates: P@5, MRR, NDCG@5, latency percentiles.

import type { EvalExpectation, EvalResult, EvalDetail } from './types.ts';
import { search } from './searcher.ts';

/** Default eval queries. These target the retrieval-engine concepts that
 *  exist in any indexed codebase of this project, so the suite produces
 *  meaningful numbers out of the box. Swap in queries that match YOUR
 *  corpus for a real benchmark. */
export const DEFAULT_EVAL_QUERIES: EvalExpectation[] = [
  { query: 'embedding provider', expected_file: 'embedding' },
  { query: 'reciprocal rank fusion', expected_file: 'fusion' },
  { query: 'inverted index posting list', expected_file: 'inverted-index' },
  { query: 'bm25 scoring', expected_file: 'bm25' },
  { query: 'query intent classification', expected_file: 'scoring' },
  { query: 'search result cache', expected_file: 'cache' },
  { query: 'tokenize text', expected_file: 'tokenizer' },
  { query: 'reranker', expected_file: 'reranker' },
  { query: 'vector cosine similarity', expected_file: 'vector-search' },
  { query: 'build context index', expected_file: 'indexer' },
];

/** Compute NDCG@K for a single query. */
function ndcgAtK(rank: number | null, k: number): number {
  if (rank === null || rank > k) return 0;
  return 1 / Math.log2(rank + 1); // IDCG=1 for single relevant doc
}

/**
 * Run the evaluation suite against the initialized engine.
 * Returns P@5, MRR, NDCG@5, and latency percentiles.
 */
export async function runEvaluation(
  queries: EvalExpectation[] = DEFAULT_EVAL_QUERIES,
): Promise<EvalResult> {
  const details: EvalDetail[] = [];
  const latencies: number[] = [];
  let hits = 0;
  let reciprocalRankSum = 0;
  let ndcgSum = 0;

  for (const { query, expected_file } of queries) {
    const start = performance.now();
    const results = await search({ query, limit: 5, mode: 'hybrid' });
    const latencyMs = performance.now() - start;
    latencies.push(latencyMs);

    // Find rank of expected file (fuzzy match on path)
    let rank: number | null = null;
    for (let i = 0; i < results.length; i += 1) {
      if (results[i]!.file_path.toLowerCase().includes(expected_file.toLowerCase())) {
        rank = i + 1;
        break;
      }
    }

    const hit = rank !== null && rank <= 5;
    if (hit) {
      hits += 1;
      reciprocalRankSum += 1 / rank!;
    }
    ndcgSum += ndcgAtK(rank, 5);

    details.push({ query, expected_file, rank, hit });
  }

  const total = queries.length;
  latencies.sort((a, b) => a - b);

  return {
    total,
    precisionAt5: hits / total,
    mrr: reciprocalRankSum / total,
    ndcgAt5: ndcgSum / total,
    latencyP50Ms: latencies[Math.floor(total * 0.5)] ?? 0,
    latencyP95Ms: latencies[Math.floor(total * 0.95)] ?? 0,
    details,
  };
}
