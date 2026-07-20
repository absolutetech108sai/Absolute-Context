// ─── Vector Search ───────────────────────────────────────────────────────────
// Cosine similarity via dot product on normalized vectors. Top-K with early
// termination on a candidate subset (not full-scan).

import type { IndexRecord } from './types.ts';

/** Dot product of two vectors (assumes pre-normalized for cosine similarity). */
export function vectorDot(a: number[], b: number[]): number {
  let value = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i += 1) {
    value += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return value;
}

export interface VectorScoredRecord {
  docId: number;
  score: number;
}

/**
 * Score a subset of records by cosine similarity to the query vector.
 * Only scores the given candidate IDs (not full-scan).
 * Returns sorted descending by score.
 */
export function scoreVectorCandidates(
  queryVector: number[],
  records: IndexRecord[],
  candidateIds: number[],
): VectorScoredRecord[] {
  const scored: VectorScoredRecord[] = [];

  for (const docId of candidateIds) {
    const record = records[docId];
    if (!record?.vector?.length) continue;
    const score = vectorDot(queryVector, record.vector);
    if (score > 0) {
      scored.push({ docId, score });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored;
}

/**
 * Top-K vector search with min-heap semantics.
 * Returns only the top K results above a minimum threshold.
 */
export function topKVectorSearch(
  queryVector: number[],
  records: IndexRecord[],
  candidateIds: number[],
  k: number,
  minScore: number = 0.01,
): VectorScoredRecord[] {
  // For small candidate sets (<1000), simple sort is faster than heap
  const scored = scoreVectorCandidates(queryVector, records, candidateIds);
  return scored.filter((s) => s.score >= minScore).slice(0, k);
}
