// ─── Inverted Index ──────────────────────────────────────────────────────────
// Pre-built posting lists for O(1) BM25 term lookup. Eliminates per-query
// full-scan tokenization (17,444 regex ops → ~50 posting list lookups).

import type { IndexRecord, InvertedIndex, PostingEntry } from './types.ts';
import { tokenize, createTokenCountMap } from './tokenizer.ts';

/** Pre-computed per-record token data for fast scoring. */
export interface RecordTokenData {
  tokenCounts: Map<string, number>;
  tokenTotal: number;
  tokenArray: string[];
  tokenSet: Set<string>;
  conceptTokenSet: Set<string>;
}

/**
 * Build an inverted index from index records.
 * Called ONCE at server startup. Tokenizes each record exactly once.
 */
export function buildInvertedIndex(records: IndexRecord[]): InvertedIndex {
  const postings = new Map<string, PostingEntry[]>();
  const docLengths: number[] = new Array(records.length);
  let totalLength = 0;

  for (let docId = 0; docId < records.length; docId += 1) {
    const tokens = tokenize(records[docId]!.search_text);
    const counts = createTokenCountMap(tokens);
    const docLength = tokens.length;
    docLengths[docId] = docLength;
    totalLength += docLength;

    for (const [token, tf] of counts.entries()) {
      let postingList = postings.get(token);
      if (!postingList) {
        postingList = [];
        postings.set(token, postingList);
      }
      postingList.push({ docId, tf });
    }
  }

  // Pre-compute IDF for each term
  const docCount = records.length;
  const idf = new Map<string, number>();
  for (const [token, postingList] of postings.entries()) {
    const df = postingList.length;
    idf.set(token, Math.log(1 + (docCount - df + 0.5) / (df + 0.5)));
  }

  return {
    postings,
    idf,
    docLengths,
    avgDocLength: docCount > 0 ? totalLength / docCount : 1,
    docCount,
  };
}

/**
 * Pre-compute token data for all records (called once at startup).
 * Eliminates re-tokenization during scoring.
 */
export function precomputeTokenData(records: IndexRecord[]): RecordTokenData[] {
  return records.map((record) => {
    const tokenArray = tokenize(record.search_text);
    const tokenCounts = createTokenCountMap(tokenArray);
    const tokenTotal = tokenArray.length;
    const tokenSet = new Set(tokenArray);

    // Pre-compute concept token variants for fast bonus scoring
    const conceptTokenSet = new Set<string>();
    if (record.concept_text) {
      for (const token of tokenize(record.concept_text)) {
        conceptTokenSet.add(token);
        // Add simple plural/singular variants
        if (token.endsWith('s') && token.length > 3) conceptTokenSet.add(token.slice(0, -1));
        else if (token.length > 2) conceptTokenSet.add(`${token}s`);
      }
    }

    return { tokenCounts, tokenTotal, tokenArray, tokenSet, conceptTokenSet };
  });
}

/** splitmix32-style integer mixer. Bijective on 32-bit ints, so as the
 *  counter increments the output is uniformly distributed — unlike a raw
 *  LCG whose residues mod docCount can collapse into a short cycle. */
function mix32(x: number): number {
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * Retrieve candidate document IDs from the inverted index.
 * Returns doc IDs that match at least one query token, ranked by
 * sum of TF-IDF weights. Includes a random sample for semantic recall.
 */
export function retrieveCandidates(
  index: InvertedIndex,
  queryTokens: string[],
  maxCandidates: number = 500,
  randomSampleSize: number = 200,
): number[] {
  const candidateScores = new Map<number, number>();

  // Accumulate TF-IDF scores from posting lists
  for (const token of queryTokens) {
    const postingList = index.postings.get(token);
    if (!postingList) continue;
    const tokenIdf = index.idf.get(token) ?? 0;

    for (const entry of postingList) {
      const current = candidateScores.get(entry.docId) ?? 0;
      candidateScores.set(entry.docId, current + entry.tf * tokenIdf);
    }
  }

  // Sort by accumulated score, take top candidates
  const sorted = [...candidateScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxCandidates)
    .map(([docId]) => docId);

  // Add a deterministic pseudo-random sample for semantic recall
  // (catches docs with zero lexical overlap). mix32 on an incrementing
  // counter gives uniform residues mod docCount; the iteration cap
  // guarantees termination even in pathological cases.
  if (randomSampleSize > 0 && index.docCount > sorted.length) {
    const existing = new Set(sorted);
    let added = 0;
    const seed = queryTokens.reduce((acc, t) => acc + t.length * 31, 7);
    const maxIter = index.docCount * 4; // hard safety cap
    for (let iter = 0; iter < maxIter && added < randomSampleSize && existing.size < index.docCount; iter += 1) {
      const docId = mix32(seed + iter) % index.docCount;
      if (!existing.has(docId)) {
        existing.add(docId);
        sorted.push(docId);
        added += 1;
      }
    }
  }

  return sorted;
}
