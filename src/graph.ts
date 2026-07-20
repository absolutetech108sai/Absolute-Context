// ─── Graph Sidecar ───────────────────────────────────────────────────────────
// Knowledge graph as a boost/injection layer on top of vector+BM25 retrieval.
// Source-first policy: graph records boost source files, not replace them.

import type { QueryIntent, ScoredCandidate } from './types.ts';

export type GraphSidecarMode = 'off' | 'auto' | 'always';

const SOURCE_ORIENTED_INTENTS = new Set<QueryIntent>([
  'implementation',
  'runtime_semantic',
  'data_model',
  'documentation',
]);

export interface GraphSidecarOptions {
  mode: GraphSidecarMode;
  intent: QueryIntent;
  sourceBoost?: number;
  sourceBoostCap?: number;
  maxInjectedFiles?: number;
}

const GRAPH_SOURCE_PATTERN = /^File:\s+(.+)$/m;

/**
 * Apply graph sidecar boost to candidates.
 * Boosts source files referenced by graph records, optionally injects
 * graph records as additional candidates.
 */
export function applyGraphSidecar(
  candidates: ScoredCandidate[],
  graphRecords: ScoredCandidate[],
  options: GraphSidecarOptions,
): ScoredCandidate[] {
  const { mode, intent, sourceBoost = 0.85, sourceBoostCap = 1.2, maxInjectedFiles = 3 } = options;

  if (mode === 'off') return candidates;
  if (mode === 'auto' && !SOURCE_ORIENTED_INTENTS.has(intent)) return candidates;
  if (graphRecords.length === 0) return candidates;

  // Extract source file references from graph records
  const sourceFiles = new Map<string, number>(); // file_path → boost score
  for (const graphRecord of graphRecords) {
    const match = GRAPH_SOURCE_PATTERN.exec(graphRecord.text);
    if (match?.[1]) {
      const filePath = match[1].trim();
      sourceFiles.set(filePath, (sourceFiles.get(filePath) ?? 0) + sourceBoost);
    }
  }

  // Apply boost to existing candidates that match graph-referenced sources
  const boosted = candidates.map((candidate) => {
    const boost = sourceFiles.get(candidate.file_path);
    if (boost && candidate.kind !== 'graph') {
      const cappedBoost = Math.min(boost, sourceBoostCap);
      return {
        ...candidate,
        _score: candidate._score + cappedBoost,
        _graph_sidecar_boost: cappedBoost,
      };
    }
    return candidate;
  });

  // Inject top graph records as additional candidates (if mode=always)
  if (mode === 'always' && maxInjectedFiles > 0) {
    const existingPaths = new Set(boosted.map((c) => c.file_path));
    const injected = graphRecords
      .filter((g) => !existingPaths.has(g.file_path))
      .slice(0, maxInjectedFiles)
      .map((g) => ({ ...g, _graph_sidecar_injected: true }));
    boosted.push(...injected);
  }

  boosted.sort((a, b) => b._score - a._score);
  return boosted;
}

/**
 * Graph candidate primary policy: should graph records appear as primary
 * results or only as sidecar boosts?
 */
export function shouldExcludeGraphPrimary(
  mode: GraphSidecarMode,
  intent: QueryIntent,
  hasExplicitKindFilter: boolean,
): boolean {
  if (hasExplicitKindFilter) return false;
  if (mode === 'off') return true;
  if (!SOURCE_ORIENTED_INTENTS.has(intent)) return false;
  // In auto/always mode with source-oriented intent, exclude graph-primary
  // (graph acts as sidecar boost only)
  return true;
}
