// ─── Multi-Tier Search Cache ─────────────────────────────────────────────────
// Always-on. L1: exact-match HashMap (O(1)). L2: semantic-similarity LRU.
// No env-var gate — cache is enabled by default.

import type { SearchResult } from './types.ts';
import { vectorDot } from './vector-search.ts';

interface L1Entry {
  results: SearchResult[];
  expiresAt: number;
}

interface L2Entry {
  vector: number[];
  results: SearchResult[];
  createdAt: number;
}

export class SearchCache {
  private l1 = new Map<string, L1Entry>();
  private l2: L2Entry[] = [];
  private maxL1: number;
  private maxL2: number;
  private ttlMs: number;
  private similarityThreshold: number;

  constructor(opts: {
    maxEntries?: number;
    ttlMs?: number;
    similarityThreshold?: number;
  } = {}) {
    this.maxL1 = opts.maxEntries ?? 512;
    this.maxL2 = Math.min(128, (opts.maxEntries ?? 512) / 4);
    this.ttlMs = opts.ttlMs ?? 300_000; // 5 minutes
    this.similarityThreshold = opts.similarityThreshold ?? 0.97;
  }

  /** Build a cache key from query + options. */
  private key(query: string, optionsKey: string): string {
    return `${query.toLowerCase().trim()}::${optionsKey}`;
  }

  /** L1 exact-match lookup. Returns null on miss. */
  lookupExact(query: string, optionsKey: string): SearchResult[] | null {
    const k = this.key(query, optionsKey);
    const entry = this.l1.get(k);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.l1.delete(k);
      return null;
    }
    return entry.results;
  }

  /** L2 semantic-similarity lookup. Returns null on miss. */
  lookupSemantic(queryVector: number[] | null, optionsKey: string): SearchResult[] | null {
    if (!queryVector || queryVector.length === 0) return null;

    let bestScore = 0;
    let bestResults: SearchResult[] | null = null;

    for (const entry of this.l2) {
      if (entry.vector.length !== queryVector.length) continue;
      const similarity = vectorDot(queryVector, entry.vector);
      if (similarity >= this.similarityThreshold && similarity > bestScore) {
        bestScore = similarity;
        bestResults = entry.results;
      }
    }

    return bestResults;
  }

  /** Combined lookup: try L1 first, then L2. */
  lookup(query: string, queryVector: number[] | null, optionsKey: string): SearchResult[] | null {
    const exact = this.lookupExact(query, optionsKey);
    if (exact) return exact;
    return this.lookupSemantic(queryVector, optionsKey);
  }

  /** Store results in both tiers. */
  store(query: string, queryVector: number[] | null, optionsKey: string, results: SearchResult[]): void {
    const k = this.key(query, optionsKey);

    // L1: exact match
    this.l1.set(k, { results, expiresAt: Date.now() + this.ttlMs });
    if (this.l1.size > this.maxL1) {
      // Evict oldest entries
      const entries = [...this.l1.entries()].sort((a, b) => a[1].expiresAt - b[1].expiresAt);
      for (const [key] of entries.slice(0, Math.ceil(this.maxL1 * 0.2))) {
        this.l1.delete(key);
      }
    }

    // L2: semantic similarity
    if (queryVector && queryVector.length > 0) {
      this.l2.push({ vector: queryVector, results, createdAt: Date.now() });
      if (this.l2.length > this.maxL2) {
        this.l2 = this.l2.slice(-this.maxL2);
      }
    }
  }

  /** Invalidate all caches (call on index rebuild). */
  invalidate(): void {
    this.l1.clear();
    this.l2 = [];
  }

  /** Cache stats for health endpoint. */
  stats(): { l1Size: number; l2Size: number; maxL1: number; maxL2: number } {
    return { l1Size: this.l1.size, l2Size: this.l2.length, maxL1: this.maxL1, maxL2: this.maxL2 };
  }
}
