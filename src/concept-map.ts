// ─── Concept Map ─────────────────────────────────────────────────────────────
// Lazy-loaded concept expansion. No module-level side effects.
// Expands queries with related terms from a concept map file.

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { tokenize } from './tokenizer.ts';

export interface ConceptMapEntry {
  term: string;
  related: string[];
  filePaths: string[];
}

export interface ConceptExpansionResult {
  applied: boolean;
  expandedQuery: string;
  appendedTerms: string[];
  matchedPaths: string[];
}

let cachedConceptMap: ConceptMapEntry[] | null = null;
let cachedConceptMapDir: string | null = null;

/** Load concept map lazily (memoized per directory). */
export function getConceptMap(rootDir: string): ConceptMapEntry[] {
  if (cachedConceptMap && cachedConceptMapDir === rootDir) {
    return cachedConceptMap;
  }

  const conceptMapPath = path.join(rootDir, '.context-concept-map.json');
  if (!existsSync(conceptMapPath)) {
    cachedConceptMap = [];
    cachedConceptMapDir = rootDir;
    return cachedConceptMap;
  }

  try {
    const raw = JSON.parse(readFileSync(conceptMapPath, 'utf8')) as Record<string, unknown>;
    const entries: ConceptMapEntry[] = [];

    for (const [term, value] of Object.entries(raw)) {
      if (typeof value === 'object' && value !== null) {
        const v = value as { related?: string[]; files?: string[] };
        entries.push({
          term,
          related: v.related ?? [],
          filePaths: v.files ?? [],
        });
      }
    }

    cachedConceptMap = entries;
    cachedConceptMapDir = rootDir;
    return entries;
  } catch {
    cachedConceptMap = [];
    cachedConceptMapDir = rootDir;
    return [];
  }
}

/**
 * Expand a query using the concept map.
 * Finds matching concepts by token overlap and appends related terms.
 */
export function expandQueryWithConceptMap(
  query: string,
  rootDir: string,
  maxAppendedTerms: number = 12,
): ConceptExpansionResult {
  const conceptMap = getConceptMap(rootDir);
  if (conceptMap.length === 0) {
    return { applied: false, expandedQuery: query, appendedTerms: [], matchedPaths: [] };
  }

  const queryTokens = new Set(tokenize(query));
  const scoredEntries: Array<{ entry: ConceptMapEntry; score: number }> = [];

  for (const entry of conceptMap) {
    const termTokens = tokenize(entry.term);
    const overlap = termTokens.filter((t) => queryTokens.has(t)).length;
    if (overlap > 0) {
      scoredEntries.push({ entry, score: overlap / termTokens.length });
    }
  }

  if (scoredEntries.length === 0) {
    return { applied: false, expandedQuery: query, appendedTerms: [], matchedPaths: [] };
  }

  // Sort by relevance, take top matches
  scoredEntries.sort((a, b) => b.score - a.score);
  const topMatches = scoredEntries.slice(0, 5);

  // Collect related terms (deduplicated, capped)
  const appendedTerms: string[] = [];
  const matchedPaths: string[] = [];
  const seen = new Set<string>(queryTokens);

  for (const { entry } of topMatches) {
    for (const term of entry.related) {
      const tokens = tokenize(term);
      if (tokens.some((t) => !seen.has(t)) && appendedTerms.length < maxAppendedTerms) {
        appendedTerms.push(term);
        for (const t of tokens) seen.add(t);
      }
    }
    matchedPaths.push(...entry.filePaths);
  }

  const expandedQuery = appendedTerms.length > 0
    ? `${query} ${appendedTerms.join(' ')}`
    : query;

  return {
    applied: appendedTerms.length > 0,
    expandedQuery,
    appendedTerms,
    matchedPaths: [...new Set(matchedPaths)].slice(0, 10),
  };
}

/** Invalidate the cached concept map (call after index rebuild). */
export function invalidateConceptMap(): void {
  cachedConceptMap = null;
  cachedConceptMapDir = null;
}
