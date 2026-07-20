// ─── PlatformOS Stress Test ──────────────────────────────────────────────────
// Indexes the parent absolutetech-platform monorepo into an isolated test
// index, then runs a battery of queries across all modes with timing.

// Isolate the test index so we never touch the monorepo's own context-index/
process.env.CONTEXT_INDEX_PATH = 'Absolute-Context/test-index-platformos';
process.env.CONTEXT_EMBEDDING_PROVIDER = 'local-hash';

const MONOREPO = '..'; // parent of Absolute-Context

import { buildIndex } from './src/indexer.ts';
import { initializeEngine, search } from './src/searcher.ts';

console.log('═══ AbsoluteContext × PlatformOS stress test ═══\n');

// ─── 1. Index the monorepo ───────────────────────────────────────────────────
// Focus on actual source (apps/packages/scripts), not the doc dumps
// (audit-reports, improvements, session-logs, obsidian) that would otherwise
// produce 285K prose chunks.
console.log('▶ Indexing monorepo source (apps, packages, scripts)...');
const t0 = performance.now();
const build = await buildIndex({
  rootDir: MONOREPO,
  includeDirs: ['apps', 'packages', 'scripts'],
  maxRecords: 60000,
});
const indexMs = Math.round(performance.now() - t0);
console.log(`  ${build.recordCount} records in ${indexMs}ms`);
console.log(`  kinds:`, build.countsByKind);
if (build.warnings.length) console.log(`  ⚠ ${build.warnings.length} warnings`);

// ─── 2. Initialize engine ────────────────────────────────────────────────────
console.log('\n▶ Initializing engine (inverted index + token data)...');
const t1 = performance.now();
initializeEngine(MONOREPO);
console.log(`  ready in ${Math.round(performance.now() - t1)}ms`);

// ─── 3. Query battery ────────────────────────────────────────────────────────
interface TestCase {
  query: string;
  expectSubstring: string; // a substring that SHOULD appear in a top-3 file path
  modes?: Array<'local' | 'global' | 'hybrid'>;
}

const cases: TestCase[] = [
  { query: 'budget enforcement cost guardrail', expectSubstring: 'billing' },
  { query: 'authentication middleware better-auth session', expectSubstring: 'auth' },
  { query: 'temporal workflow chat pipeline', expectSubstring: 'workflow' },
  { query: 'DLP pii masking india gstin', expectSubstring: 'dlp' },
  { query: 'drizzle schema pgtable users', expectSubstring: 'schema' },
  { query: 'rate limit middleware', expectSubstring: 'rate-limit' },
  { query: 'idempotency store setnx', expectSubstring: 'idempotency' },
  { query: 'circuit breaker redis grace', expectSubstring: 'kernel' },
  { query: 'uncertainty propagation model selection', expectSubstring: 'intelligence' },
  { query: 'output safety guard', expectSubstring: 'guards' },
];

console.log('\n▶ Query battery (hybrid mode, top-3):');
let hits = 0;
const latencies: number[] = [];

for (const { query, expectSubstring } of cases) {
  const t = performance.now();
  const results = await search({ rootDir: MONOREPO, query, mode: 'hybrid', limit: 3 });
  const ms = Math.round(performance.now() - t);
  latencies.push(ms);

  const topFiles = results.map((r) => r.file_path);
  const hit = topFiles.some((f) => f.toLowerCase().includes(expectSubstring.toLowerCase()));
  if (hit) hits += 1;

  console.log(`  ${hit ? '✓' : '✗'} [${String(ms).padStart(4)}ms] "${query}"`);
  topFiles.forEach((f, i) => console.log(`      ${i + 1}. ${f}`));
}

// ─── 4. Mode comparison on one query ────────────────────────────────────────
console.log('\n▶ Mode comparison: "budget enforcement"');
for (const mode of ['local', 'global', 'hybrid'] as const) {
  const t = performance.now();
  const results = await search({ rootDir: MONOREPO, query: 'budget enforcement', mode, limit: 3 });
  const ms = Math.round(performance.now() - t);
  console.log(`  [${mode.padEnd(6)} ${String(ms).padStart(4)}ms] ${results.map((r) => r.file_path).join(' | ')}`);
}

// ─── 5. Cache behavior ───────────────────────────────────────────────────────
console.log('\n▶ Cache behavior (repeat query):');
await search({ rootDir: MONOREPO, query: 'budget enforcement', mode: 'hybrid', limit: 3 });
const tc = performance.now();
await search({ rootDir: MONOREPO, query: 'budget enforcement', mode: 'hybrid', limit: 3 });
console.log(`  cached repeat: ${Math.round(performance.now() - tc)}ms`);

// ─── 6. Summary ──────────────────────────────────────────────────────────────
latencies.sort((a, b) => a - b);
console.log('\n═══ Summary ═══');
console.log(`  Records indexed:  ${build.recordCount}`);
console.log(`  Index build time: ${indexMs}ms`);
console.log(`  Query hit rate:   ${hits}/${cases.length} (${Math.round((100 * hits) / cases.length)}%)`);
console.log(`  Latency P50:      ${latencies[Math.floor(latencies.length * 0.5)]}ms`);
console.log(`  Latency P95:      ${latencies[Math.floor(latencies.length * 0.95)]}ms`);
