// ─── Reciprocal Rank Fusion (RRF) ───────────────────────────────────────────
// Distribution-agnostic rank fusion. Replaces brittle min-max weighted sum.
// Proven in TREC evaluations since 2009 (Cormack et al., k=60 standard).

import type { ScoredCandidate } from './types.ts';

export interface RankedItem {
  docId: number;
  rank: number;
}

/**
 * Reciprocal Rank Fusion: merge multiple ranked lists into one.
 * RRF_score(d) = Σ 1/(k + rank_i(d)) for each ranker i.
 *
 * @param rankings - Array of ranked lists (each sorted by score desc)
 * @param k - Smoothing constant (default 60, standard from literature)
 * @returns Fused candidates sorted by RRF score descending
 */
export function reciprocalRankFusion(
  rankings: RankedItem[][],
  k: number = 60,
): Map<number, number> {
  const scores = new Map<number, number>();

  for (const ranking of rankings) {
    for (let rank = 0; rank < ranking.length; rank += 1) {
      const item = ranking[rank]!;
      const current = scores.get(item.docId) ?? 0;
      scores.set(item.docId, current + 1 / (k + rank + 1));
    }
  }

  return scores;
}

/**
 * Apply RRF to BM25 and vector ranked lists, returning fused candidates.
 * Preserves the original candidate data while adding _rrf_score.
 */
export function fuseWithRRF(
  bm25Ranked: ScoredCandidate[],
  vectorRanked: ScoredCandidate[],
  candidates: ScoredCandidate[],
  k: number = 60,
): ScoredCandidate[] {
  // Build rank lists
  const bm25Ranks: RankedItem[] = bm25Ranked.map((c, i) => ({ docId: c.docId, rank: i }));
  const vectorRanks: RankedItem[] = vectorRanked.map((c, i) => ({ docId: c.docId, rank: i }));

  const rrfScores = reciprocalRankFusion([bm25Ranks, vectorRanks], k);

  // Apply RRF scores to candidates
  const fused = candidates.map((candidate) => ({
    ...candidate,
    _rrf_score: rrfScores.get(candidate.docId) ?? 0,
    _score: rrfScores.get(candidate.docId) ?? 0,
  }));

  fused.sort((a, b) => (b._rrf_score ?? 0) - (a._rrf_score ?? 0));
  return fused;
}

/**
 * Legacy weighted-sum fusion (kept for A/B comparison).
 * score = semanticWeight × semantic_norm + lexicalWeight × lexical_norm
 */
export function weightedSumFusion(
  candidates: ScoredCandidate[],
  semanticWeight: number = 0.55,
  lexicalWeight: number = 0.45,
  ftsOnly: boolean = false,
): ScoredCandidate[] {
  const scored = candidates.map((candidate) => {
    const score = ftsOnly
      ? (candidate._lexical_normalized ?? 0)
      : semanticWeight * (candidate._semantic_normalized ?? 0) +
        lexicalWeight * (candidate._lexical_normalized ?? 0);
    return { ...candidate, _score: score };
  });

  scored.sort((a, b) => b._score - a._score);
  return scored;
}
