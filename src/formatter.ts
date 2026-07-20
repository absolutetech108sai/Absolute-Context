// ─── Formatter ───────────────────────────────────────────────────────────────
// Tiered output formatting: brief (~50 tokens), normal (~500), full (all text).

import type { SearchResult } from './types.ts';

export type OutputMode = 'brief' | 'normal' | 'full' | 'json';

/** Format results for CLI/API output at the specified verbosity tier. */
export function formatResults(results: SearchResult[], mode: OutputMode = 'normal'): string {
  if (mode === 'json') {
    return JSON.stringify(results, null, 2);
  }

  if (mode === 'brief') {
    return results
      .map((r) => {
        const sym = r.symbols ? ` [${r.symbols}]` : '';
        return `${r.file_path} (${r.kind})${sym} score=${r._score.toFixed(2)}`;
      })
      .join('\n');
  }

  if (mode === 'full') {
    return results
      .map((r) => {
        const header = `--- ${r.file_path} [${r.kind}] L${r.line_start}-${r.line_end} score=${r._score.toFixed(2)} ---`;
        const symbols = r.symbols ? `Symbols: ${r.symbols}` : '';
        const heading = r.heading ? `Heading: ${r.heading}` : '';
        return [header, symbols, heading, '', r.text].filter(Boolean).join('\n');
      })
      .join('\n\n');
  }

  // Normal mode: path + score + first 3 lines of text
  return results
    .map((r) => {
      const sym = r.symbols ? ` [${r.symbols}]` : '';
      const preview = r.text.split('\n').slice(0, 3).join('\n  ');
      return `${r.file_path}:${r.line_start}-${r.line_end} (${r.kind})${sym} score=${r._score.toFixed(2)}\n  ${preview}`;
    })
    .join('\n\n');
}
