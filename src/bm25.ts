// ─── BM25 Scoring ────────────────────────────────────────────────────────────
// Okapi BM25 with phrase bonus, bigram bonus, coverage bonus, and symbol bonus.

import type { IndexRecord } from './types.ts';
import {
  tokenize,
  createTokenCountMap,
  containsOrderedTokenSequence,
  bigrams,
} from './tokenizer.ts';

export interface BM25Params {
  k1: number;
  b: number;
}

export const DEFAULT_BM25_PARAMS: BM25Params = { k1: 1.2, b: 0.75 };

/**
 * Compute BM25 score for a single record against query tokens.
 * Includes phrase adjacency bonus (+2), bigram bonus (≤0.75),
 * coverage bonus (+0.5), and symbol match bonus (+0.5).
 */
export function bm25Score(
  queryTokens: string[],
  record: IndexRecord,
  docFrequency: Record<string, number>,
  averageLength: number,
  documentCount: number,
  params: BM25Params = DEFAULT_BM25_PARAMS,
): number {
  const tokenCounts = createTokenCountMap(tokenize(record.search_text));
  const tokenTotal = Array.from(tokenCounts.values()).reduce((sum, v) => sum + v, 0);
  const { k1, b } = params;
  const docLength = Math.max(1, tokenTotal);
  let score = 0;

  // Core BM25 TF-IDF scoring
  for (const token of queryTokens) {
    const tf = tokenCounts.get(token) ?? 0;
    if (!tf) continue;
    const df = docFrequency[token] ?? 0;
    const idf = Math.log(1 + (documentCount - df + 0.5) / (df + 0.5));
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + (b * docLength) / Math.max(1, averageLength));
    score += idf * (numerator / denominator);
  }

  // Phrase bonus: rewards adjacent ordered token sequences
  const haystackTokens = tokenize(record.search_text);
  const haystackTokenSet = new Set(haystackTokens);
  const significantTokens = queryTokens.filter((t) => t.length >= 3);

  if (significantTokens.length > 1 && containsOrderedTokenSequence(haystackTokens, significantTokens)) {
    score += 2;
  } else if (significantTokens.length > 1) {
    // Bigram partial match bonus
    const haystackBigrams = new Set(bigrams(haystackTokens));
    const queryBigrams = bigrams(significantTokens);
    const matches = queryBigrams.filter((pair) => haystackBigrams.has(pair)).length;
    if (matches > 0) {
      score += Math.min(0.75, matches * 0.25);
    }
  }

  // Coverage bonus: all significant query terms present (any order)
  if (significantTokens.length > 0 && significantTokens.every((t) => haystackTokenSet.has(t))) {
    score += 0.5;
  }

  // Symbol match bonus: query terms appear in exported symbols
  if (record.symbols && significantTokens.length > 0) {
    const symbolLower = record.symbols.toLowerCase();
    if (significantTokens.some((t) => symbolLower.includes(t))) {
      score += 0.5;
    }
  }

  return score;
}

/**
 * BM25 scoring using pre-computed token data (for inverted index path).
 * Avoids re-tokenizing the record's search_text.
 */
export function bm25ScorePrecomputed(
  queryTokens: string[],
  tokenCounts: Map<string, number>,
  docLength: number,
  docFrequency: Record<string, number>,
  averageLength: number,
  documentCount: number,
  params: BM25Params = DEFAULT_BM25_PARAMS,
): number {
  const { k1, b } = params;
  const safeDocLength = Math.max(1, docLength);
  let score = 0;

  for (const token of queryTokens) {
    const tf = tokenCounts.get(token) ?? 0;
    if (!tf) continue;
    const df = docFrequency[token] ?? 0;
    const idf = Math.log(1 + (documentCount - df + 0.5) / (df + 0.5));
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + (b * safeDocLength) / Math.max(1, averageLength));
    score += idf * (numerator / denominator);
  }

  return score;
}

/** Min-max normalize a numeric field across a result set. */
export function normalizeScores<T extends Record<string, unknown>>(
  results: T[],
  field: string,
): T[] {
  const values = results.map((r) => Number(r[field] ?? 0));
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) {
    return results.map((r) => ({
      ...r,
      [`${field}_normalized`]: Number(r[field] ?? 0) > 0 ? 1 : 0,
    }));
  }
  return results.map((r) => ({
    ...r,
    [`${field}_normalized`]: (Number(r[field] ?? 0) - min) / (max - min),
  }));
}
