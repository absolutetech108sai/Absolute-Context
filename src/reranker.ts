// ─── Reranker ────────────────────────────────────────────────────────────────
// GPT-4o-mini cross-encoder reranker. Batches top-K candidates into one call.
// Graceful degradation: on timeout/failure, returns original order.

import type { Reranker, ScoredCandidate } from './types.ts';

const registry = new Map<string, () => Reranker>();

/** Register a custom reranker factory. */
export function registerReranker(name: string, factory: () => Reranker): void {
  registry.set(name, factory);
}

/** No-op reranker — returns candidates unchanged. */
export class NoopReranker implements Reranker {
  readonly name = 'noop';
  async rerank(_query: string, candidates: ScoredCandidate[], _limit: number): Promise<ScoredCandidate[]> {
    return candidates;
  }
}

/** GPT-4o-mini based reranker. Scores relevance 0-10 for each candidate. */
export class OpenAIReranker implements Reranker {
  readonly name = 'openai';
  private model: string;
  private timeoutMs: number;
  private apiKey: string;

  constructor(opts: { model?: string; timeoutMs?: number; apiKey?: string } = {}) {
    this.model = opts.model ?? 'gpt-4o-mini';
    this.timeoutMs = opts.timeoutMs ?? 3000;
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? '';
  }

  async rerank(query: string, candidates: ScoredCandidate[], limit: number): Promise<ScoredCandidate[]> {
    if (!this.apiKey || candidates.length === 0) return candidates;

    const topCandidates = candidates.slice(0, Math.min(15, candidates.length));

    try {
      const snippets = topCandidates.map((c, i) => {
        const snippet = c.text.slice(0, 200).replace(/\n/g, ' ');
        return `[${i}] ${c.file_path}:${c.line_start}-${c.line_end} (${c.kind})\n${snippet}`;
      }).join('\n\n');

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          messages: [{
            role: 'user',
            content: `Rate the relevance of each code snippet to the query "${query}" on a scale of 0-10. Return ONLY a JSON array of numbers in the same order as the snippets.\n\n${snippets}`,
          }],
          max_tokens: 100,
          temperature: 0,
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) return candidates;

      const payload = (await response.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };

      const content = payload.choices?.[0]?.message?.content ?? '';
      const scores = JSON.parse(content.replace(/[^[\d,.\-\s]]/g, '')) as number[];

      if (!Array.isArray(scores) || scores.length !== topCandidates.length) {
        return candidates;
      }

      // Apply reranker scores and re-sort
      const reranked = topCandidates.map((c, i) => ({
        ...c,
        _rerank_score: scores[i] ?? 0,
        _score: (scores[i] ?? 0) / 10, // Normalize to 0-1
      }));

      reranked.sort((a, b) => (b._rerank_score ?? 0) - (a._rerank_score ?? 0));

      // Append remaining candidates that weren't reranked
      const rerankedIds = new Set(reranked.map((c) => c.docId));
      const remaining = candidates.filter((c) => !rerankedIds.has(c.docId));

      return [...reranked, ...remaining].slice(0, limit);
    } catch {
      // Timeout or parse failure — graceful degradation
      return candidates.slice(0, limit);
    }
  }
}

/** Create a reranker from config. */
export function createReranker(config: { reranker: string; reranker_model?: string; reranker_timeout_ms?: number }): Reranker {
  const custom = registry.get(config.reranker);
  if (custom) return custom();

  switch (config.reranker) {
    case 'openai':
      if (!process.env.OPENAI_API_KEY) return new NoopReranker();
      return new OpenAIReranker({
        model: config.reranker_model,
        timeoutMs: config.reranker_timeout_ms,
      });
    default:
      return new NoopReranker();
  }
}
