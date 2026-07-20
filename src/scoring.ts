// ─── Scoring ─────────────────────────────────────────────────────────────────
// Intent-aware multi-signal bonus scoring. 20+ tunable weights that adjust
// retrieval ranking based on query intent and record characteristics.

import type { QueryIntent, ScoredCandidate } from './types.ts';
import { tokenize, squashIdentifier, conceptTokenVariants } from './tokenizer.ts';
import type { RecordTokenData } from './inverted-index.ts';

// ─── Intent Classification ───────────────────────────────────────────────────

export function inferQueryIntent(query: string): QueryIntent {
  const lower = query.toLowerCase();

  if (/^(feat|fix|docs|style|revert|build|ci|perf|deps|test|tests|chore|refactor)(?:\([^)]*\))?!?:/.test(lower)) {
    return 'commit_message';
  }
  if (/\b(financial audit trail|chat api calls?|per-conversation spend|chat session costs?)\b/.test(lower)) {
    return 'runtime_semantic';
  }
  if (/(money model|schema context|execution log|agent context|readme|audit|plan|guide|prd)/.test(lower)) {
    return 'documentation';
  }
  if (
    /\b(table|tables|schema|column|columns|relation|relations|pgtable|migration|migrations|sql)\b/.test(lower) ||
    /\b(authentication state|auth state|sessions|accounts|verifications)\b/.test(lower)
  ) {
    return 'data_model';
  }
  if (
    /^(how|why|what|where|when|which|show|explain|describe)\b/.test(lower) ||
    /(guarantee|prevent|ensure|track|store|implement|handle|manage)/.test(lower)
  ) {
    return 'runtime_semantic';
  }
  return 'implementation';
}

// ─── Bonus Weights ───────────────────────────────────────────────────────────

export interface ResultBonusWeights {
  pathMatch: number;
  rawTextMatch: number;
  exactSymbolMatch: number;
  partialSymbolMatch: number;
  conceptTokenMatch: number;
  conceptCap: number;
  toolingPenalty: number;
  commitMessageToolingBonus: number;
  recencyBucket0: number;
  recencyBucket1: number;
  implementationCode: number;
  implementationGraph: number;
  implementationDocPenalty: number;
  runtimeCode: number;
  runtimeSchema: number;
  runtimeNonRuntimePenalty: number;
  dataModelSchema: number;
  dataModelCode: number;
  dataModelDoc: number;
  dataModelGraphPenalty: number;
  documentationDoc: number;
}

export const DEFAULT_RESULT_BONUS_WEIGHTS: ResultBonusWeights = {
  pathMatch: 1.75,
  rawTextMatch: 1.25,
  exactSymbolMatch: 3,
  partialSymbolMatch: 1.5,
  conceptTokenMatch: 0.2,
  conceptCap: 0.75,
  toolingPenalty: -0.35,
  commitMessageToolingBonus: 0.1,
  recencyBucket0: 0.15,
  recencyBucket1: 0.05,
  implementationCode: 0.45,
  implementationGraph: -0.45,
  implementationDocPenalty: -0.1,
  runtimeCode: 0.45,
  runtimeSchema: 0.15,
  runtimeNonRuntimePenalty: -0.1,
  dataModelSchema: 0.5,
  dataModelCode: 0.15,
  dataModelDoc: 0.1,
  dataModelGraphPenalty: -0.05,
  documentationDoc: 0.25,
};

// ─── Bonus Computation ───────────────────────────────────────────────────────

/**
 * Compute the additive bonus for a candidate based on query, intent,
 * and record characteristics. Uses pre-computed token data when available.
 */
export function resultBonus(
  query: string,
  candidate: ScoredCandidate,
  intent: QueryIntent,
  tokenData?: RecordTokenData,
  weights: Partial<ResultBonusWeights> = {},
): number {
  const w = { ...DEFAULT_RESULT_BONUS_WEIGHTS, ...weights };
  const rawQuery = query.toLowerCase().trim();
  const squashedQuery = squashIdentifier(query);
  const filePath = candidate.file_path.toLowerCase();
  const baseName = filePath.split('/').pop()?.replace(/\.[^.]+$/, '') ?? '';
  const squashedBase = squashIdentifier(baseName);
  const symbolList = (candidate.symbols ?? '').split(',').map((s) => s.trim()).filter(Boolean);

  let bonus = 0;

  // Path match: query appears in file path
  if (filePath.includes(rawQuery) || squashedBase.includes(squashedQuery)) {
    bonus += w.pathMatch;
  }

  // Raw text match: query appears verbatim in search text
  if (candidate.search_text.toLowerCase().includes(rawQuery)) {
    bonus += w.rawTextMatch;
  }

  // Symbol match: exact or partial
  if (symbolList.some((s) => squashIdentifier(s) === squashedQuery)) {
    bonus += w.exactSymbolMatch;
  } else if (symbolList.some((s) => squashIdentifier(s).includes(squashedQuery))) {
    bonus += w.partialSymbolMatch;
  }

  // Concept token match (uses pre-computed set when available)
  const queryTokens = tokenize(rawQuery).filter((t) => t.length >= 3);
  if (tokenData?.conceptTokenSet.size) {
    const matches = queryTokens.filter((t) =>
      conceptTokenVariants(t).some((v) => tokenData.conceptTokenSet.has(v)),
    ).length;
    if (matches > 0) {
      bonus += Math.min(w.conceptCap, matches * w.conceptTokenMatch);
    }
  } else if (candidate.concept_text) {
    const conceptTokens = new Set(tokenize(candidate.concept_text).flatMap(conceptTokenVariants));
    const matches = queryTokens.filter((t) =>
      conceptTokenVariants(t).some((v) => conceptTokens.has(v)),
    ).length;
    if (matches > 0) {
      bonus += Math.min(w.conceptCap, matches * w.conceptTokenMatch);
    }
  }

  // Tooling penalty/bonus
  if (candidate.kind === 'tooling') {
    bonus += intent === 'commit_message' ? w.commitMessageToolingBonus : w.toolingPenalty;
  }

  // Recency bonus
  if (candidate.recency_bucket === 0) bonus += w.recencyBucket0;
  else if (candidate.recency_bucket === 1) bonus += w.recencyBucket1;

  // Intent-specific kind bonuses
  if (intent === 'implementation') {
    if (candidate.kind === 'code' || candidate.kind === 'schema') bonus += w.implementationCode;
    if (candidate.kind === 'graph') bonus += w.implementationGraph;
    if (candidate.kind === 'doc' || candidate.kind === 'conversation') bonus += w.implementationDocPenalty;
  } else if (intent === 'commit_message') {
    if (candidate.kind === 'code' || candidate.kind === 'schema' || candidate.kind === 'tooling') {
      bonus += w.implementationCode * 0.5;
    }
  } else if (intent === 'runtime_semantic') {
    if (candidate.kind === 'code') bonus += w.runtimeCode;
    if (candidate.kind === 'schema') bonus += w.runtimeSchema;
    if (candidate.kind === 'doc' || candidate.kind === 'conversation' || candidate.kind === 'graph') {
      bonus += w.runtimeNonRuntimePenalty;
    }
  } else if (intent === 'data_model') {
    if (candidate.kind === 'schema') bonus += w.dataModelSchema;
    if (candidate.kind === 'code') bonus += w.dataModelCode;
    if (candidate.kind === 'doc' || candidate.kind === 'agent_rules') bonus += w.dataModelDoc;
    if (candidate.kind === 'graph') bonus += w.dataModelGraphPenalty;
  } else if (candidate.kind === 'doc' || candidate.kind === 'conversation' || candidate.kind === 'agent_rules') {
    bonus += w.documentationDoc;
  }

  return bonus;
}
