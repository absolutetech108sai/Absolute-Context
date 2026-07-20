// ─── CodeContext CLI ─────────────────────────────────────────────────────────
// Usage:
//   node src/cli.ts index [--root DIR]
//   node src/cli.ts search "query" [--mode hybrid|local|global] [--limit N]
//   node src/cli.ts eval

import { initializeEngine, search, getEngineHealth } from './searcher.ts';
import { buildIndex } from './indexer.ts';
import { runEvaluation } from './eval.ts';
import { formatResults } from './formatter.ts';
import type { QueryMode } from './types.ts';

const args = process.argv.slice(2);
const command = args[0] ?? 'help';

async function main() {
  switch (command) {
    case 'index': {
      const rootFlag = args.indexOf('--root');
      const rootDir = rootFlag !== -1 ? args[rootFlag + 1] : process.cwd();
      console.log(`[codecontext] Building index from ${rootDir}...`);
      const result = await buildIndex({ rootDir });
      console.log(`[codecontext] Indexed ${result.recordCount} records in ${result.durationMs}ms`);
      console.log(`[codecontext] Kinds:`, result.countsByKind);
      if (result.warnings.length > 0) {
        console.warn(`[codecontext] ${result.warnings.length} warnings`);
      }
      break;
    }

    case 'search': {
      const query = args[1];
      if (!query) {
        console.error('Usage: node src/cli.ts search "query" [--mode hybrid|local|global] [--limit N] [--rerank]');
        process.exit(1);
      }
      const modeFlag = args.indexOf('--mode');
      const limitFlag = args.indexOf('--limit');
      const mode = (modeFlag !== -1 ? args[modeFlag + 1] : 'hybrid') as QueryMode;
      const limit = limitFlag !== -1 ? Number(args[limitFlag + 1]) : 10;
      const rerank = args.includes('--rerank');

      initializeEngine(process.cwd());
      const results = await search({ query, mode, limit, rerank });
      console.log(formatResults(results, 'normal'));
      console.log(`\n--- ${results.length} results (mode=${mode}, rerank=${rerank}) ---`);
      break;
    }

    case 'eval': {
      initializeEngine(process.cwd());
      console.log('[codecontext] Running evaluation...');
      const result = await runEvaluation();
      console.log(`\n[codecontext] Eval Results:`);
      console.log(`  P@5:       ${result.precisionAt5.toFixed(3)}`);
      console.log(`  MRR:       ${result.mrr.toFixed(3)}`);
      console.log(`  NDCG@5:    ${result.ndcgAt5.toFixed(3)}`);
      console.log(`  Latency P50: ${result.latencyP50Ms.toFixed(1)}ms`);
      console.log(`  Latency P95: ${result.latencyP95Ms.toFixed(1)}ms`);
      console.log(`\n  Details:`);
      for (const d of result.details) {
        const status = d.hit ? '✓' : '✗';
        console.log(`    ${status} "${d.query}" → rank=${d.rank ?? 'miss'} (expected: ${d.expected_file})`);
      }
      break;
    }

    case 'health': {
      initializeEngine(process.cwd());
      console.log(JSON.stringify(getEngineHealth(), null, 2));
      break;
    }

    default: {
      console.log(`
CodeContext — Intent-aware code retrieval engine

Commands:
  index   [--root DIR]              Build the context index
  search  "query" [options]         Search the index
  eval                              Run evaluation suite
  health                            Show engine health

Search options:
  --mode hybrid|local|global        Retrieval mode (default: hybrid)
  --limit N                         Max results (default: 10)
  --rerank                          Enable GPT-4o-mini reranker

Examples:
  node src/cli.ts index --root /path/to/repo
  node src/cli.ts search "authentication middleware" --mode hybrid
  node src/cli.ts search "schema migration" --rerank --limit 5
  node src/cli.ts eval
`);
    }
  }
}

main().catch((err) => {
  console.error('[codecontext] Fatal:', err);
  process.exit(1);
});
