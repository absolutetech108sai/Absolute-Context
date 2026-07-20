// ─── Query Pipeline ──────────────────────────────────────────────────────────
// Pre-retrieval query understanding: intent classification, decomposition,
// and rewriting. GPT-powered when available, regex fallback offline.

import type { ContextConfig, ProcessedQuery, QueryIntent, QueryPipeline } from './types.ts';
import { inferQueryIntent } from './scoring.ts';
import { tokenize } from './tokenizer.ts';

// ─── Regex Pipeline (offline fallback) ───────────────────────────────────────

export class RegexQueryPipeline implements QueryPipeline {
  async process(query: string, _config: ContextConfig): Promise<ProcessedQuery> {
    const intent = inferQueryIntent(query);
    return {
      original: query,
      subQueries: [query],
      intent,
      expandedTerms: [],
    };
  }
}

// ─── GPT Pipeline (full query understanding) ─────────────────────────────────

const CONJUNCTION_PATTERN = /\b(and|also|as well as|plus|additionally)\b/i;
const MULTI_PART_PATTERN = /[;]|(?:\d+\.\s)/;

function shouldDecompose(query: string): boolean {
  const tokens = tokenize(query);
  if (tokens.length < 8) return false;
  return CONJUNCTION_PATTERN.test(query) || MULTI_PART_PATTERN.test(query);
}

export class GPTQueryPipeline implements QueryPipeline {
  private apiKey: string;
  private model: string;
  private responseCache = new Map<string, { result: ProcessedQuery; expiresAt: number }>();

  constructor(opts: { apiKey?: string; model?: string } = {}) {
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? '';
    this.model = opts.model ?? 'gpt-4o-mini';
  }

  async process(query: string, config: ContextConfig): Promise<ProcessedQuery> {
    // Fast path: regex intent (always runs, <1ms)
    const intent = inferQueryIntent(query);

    // Check cache
    const cacheKey = `${query}:${config.query_decomposition}`;
    const cached = this.responseCache.get(cacheKey);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.result;
    }

    // Skip GPT for short queries or when disabled
    const tokens = tokenize(query);
    if (tokens.length < 4 || config.query_understanding === 'regex' || config.query_understanding === 'off') {
      return { original: query, subQueries: [query], intent, expandedTerms: [] };
    }

    // Decomposition: only for multi-part queries in auto mode
    let subQueries = [query];
    if (
      (config.query_decomposition === 'always' || (config.query_decomposition === 'auto' && shouldDecompose(query))) &&
      this.apiKey
    ) {
      subQueries = await this.decompose(query);
    }

    const result: ProcessedQuery = {
      original: query,
      subQueries,
      intent,
      expandedTerms: [],
    };

    // Cache for 1 hour
    this.responseCache.set(cacheKey, { result, expiresAt: Date.now() + 3600_000 });
    if (this.responseCache.size > 256) {
      // Evict oldest entries
      const entries = [...this.responseCache.entries()].sort((a, b) => a[1].expiresAt - b[1].expiresAt);
      for (const [key] of entries.slice(0, 64)) {
        this.responseCache.delete(key);
      }
    }

    return result;
  }

  private async decompose(query: string): Promise<string[]> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2000);

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
            content: `Split this code search query into independent sub-queries. Return ONLY a JSON array of strings. Max 3 sub-queries, each ≤20 tokens.\n\nQuery: "${query}"`,
          }],
          max_tokens: 150,
          temperature: 0,
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);
      if (!response.ok) return [query];

      const payload = (await response.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };

      const content = payload.choices?.[0]?.message?.content ?? '';
      const parsed = JSON.parse(content.replace(/^[^[]*/, '').replace(/[^]]*$/, '')) as string[];

      if (!Array.isArray(parsed) || parsed.length === 0) return [query];

      // Validate sub-queries
      const valid = parsed
        .filter((sq) => typeof sq === 'string' && tokenize(sq).length >= 2 && tokenize(sq).length <= 20)
        .slice(0, 3);

      return valid.length > 0 ? valid : [query];
    } catch {
      return [query];
    }
  }
}

/** Create the appropriate query pipeline from config. */
export function createQueryPipeline(config: ContextConfig): QueryPipeline {
  if (config.query_understanding === 'off' || config.query_understanding === 'regex') {
    return new RegexQueryPipeline();
  }
  if (!process.env.OPENAI_API_KEY) {
    return new RegexQueryPipeline();
  }
  return new GPTQueryPipeline();
}
