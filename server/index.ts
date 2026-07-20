// ─── CodeContext Server ──────────────────────────────────────────────────────
// Hono HTTP API wrapping the retrieval engine. Serves the WebUI static build.

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { serve } from '@hono/node-server';
import { serveStatic } from '@hono/node-server/serve-static';
import { initializeEngine, search, getEngineHealth } from '../src/searcher.ts';
import { buildIndex } from '../src/indexer.ts';
import { runEvaluation } from '../src/eval.ts';
import type { QueryMode } from '../src/types.ts';

const PORT = Number(process.env.PORT ?? 3000);
const ROOT_DIR = process.env.CODECONTEXT_ROOT ?? process.cwd();

// Initialize engine at startup (pre-loads index + builds inverted index)
console.log(`[codecontext] Initializing engine from ${ROOT_DIR}...`);
const startTime = performance.now();
initializeEngine(ROOT_DIR);
console.log(`[codecontext] Engine ready in ${Math.round(performance.now() - startTime)}ms`);

const app = new Hono();

// CORS for development
app.use('/*', cors());

// ─── Search Endpoint ─────────────────────────────────────────────────────────

app.get('/api/search', async (c) => {
  const query = c.req.query('q') ?? c.req.query('query') ?? '';
  if (!query) {
    return c.json({ error: 'Query parameter "q" is required' }, 400);
  }

  const mode = (c.req.query('mode') ?? 'hybrid') as QueryMode;
  const limit = Number(c.req.query('limit') ?? 10);
  const kind = c.req.query('kind') ?? undefined;
  const rerank = c.req.query('rerank') === 'true';

  const start = performance.now();
  try {
    const results = await search({
      rootDir: ROOT_DIR,
      query,
      mode,
      limit,
      kindFilter: kind,
      rerank,
    });
    const latencyMs = Math.round(performance.now() - start);

    return c.json({
      results: results.map((r) => ({
        file_path: r.file_path,
        line_start: r.line_start,
        line_end: r.line_end,
        kind: r.kind,
        symbols: r.symbols,
        heading: r.heading,
        score: Number(r._score.toFixed(4)),
        citation: r.citation ?? `${r.file_path}:${r.line_start}-${r.line_end}`,
        snippet: r.text.slice(0, 500),
        backend: r._backend,
      })),
      meta: {
        query,
        mode,
        limit,
        latency_ms: latencyMs,
        result_count: results.length,
        reranked: rerank,
      },
    });
  } catch (err) {
    return c.json({ error: String(err) }, 500);
  }
});

// ─── Health Endpoint ─────────────────────────────────────────────────────────

app.get('/api/health', (c) => {
  return c.json(getEngineHealth());
});

// ─── Reindex Endpoint ────────────────────────────────────────────────────────

app.post('/api/reindex', async (c) => {
  try {
    const result = await buildIndex({ rootDir: ROOT_DIR, incremental: true });
    // Re-initialize engine with new index
    initializeEngine(ROOT_DIR);
    return c.json(result);
  } catch (err) {
    return c.json({ error: String(err) }, 500);
  }
});

// ─── Eval Endpoint ───────────────────────────────────────────────────────────

app.get('/api/eval', async (c) => {
  try {
    const result = await runEvaluation();
    return c.json(result);
  } catch (err) {
    return c.json({ error: String(err) }, 500);
  }
});

// ─── Static WebUI ────────────────────────────────────────────────────────────

app.use('/*', serveStatic({ root: './web/dist' }));
app.get('*', serveStatic({ path: './web/dist/index.html' }));

// ─── Start ───────────────────────────────────────────────────────────────────

serve({ fetch: app.fetch, port: PORT }, (info) => {
  console.log(`[codecontext] Server running at http://localhost:${info.port}`);
  console.log(`[codecontext] API: http://localhost:${info.port}/api/search?q=your+query`);
});
