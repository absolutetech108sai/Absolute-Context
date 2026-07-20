// ─── Indexer ─────────────────────────────────────────────────────────────────
// Builds the context index from source files. Walks the repo, chunks files,
// embeds chunks, and serializes to JSONL + manifest.

import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import type { BuildIndexOptions, BuildIndexResult, ContextConfig, IndexRecord } from './types.ts';
import { loadConfig, resolveIndexPath } from './config.ts';
import { createEmbeddingProvider } from './embeddings/index.ts';
import { estimateTokens, tokenize, createTokenCountMap } from './tokenizer.ts';

const IGNORED_DIRS = new Set([
  'node_modules', '.git', 'dist', 'build', '.cache', '.turbo',
  'coverage', '.next', 'out', 'graphify-out', 'context-index',
]);

const CODE_EXTENSIONS = new Set([
  '.ts', '.tsx', '.mts', '.js', '.mjs', '.jsx', '.py', '.go', '.rs', '.java', '.c', '.h', '.cpp',
]);

const DOC_EXTENSIONS = new Set(['.md', '.mdx', '.txt', '.rst']);
const SCHEMA_EXTENSIONS = new Set(['.sql', '.prisma']);

function classifyFile(filePath: string): IndexRecord['kind'] {
  const ext = path.extname(filePath).toLowerCase();
  const base = path.basename(filePath).toLowerCase();
  if (CODE_EXTENSIONS.has(ext)) {
    if (base.includes('.test.') || base.includes('.spec.')) return 'test';
    if (filePath.includes('schema')) return 'schema';
    return 'code';
  }
  if (DOC_EXTENSIONS.has(ext)) return 'doc';
  if (SCHEMA_EXTENSIONS.has(ext)) return 'schema';
  if (base === 'agents.md' || base.includes('.agent')) return 'agent_rules';
  return 'tooling';
}

function chunkText(text: string, maxTokens: number, overlap: number): Array<{ text: string; lineStart: number; lineEnd: number }> {
  const lines = text.split('\n');
  const chunks: Array<{ text: string; lineStart: number; lineEnd: number }> = [];
  let current: string[] = [];
  let currentTokens = 0;
  let lineStart = 1;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!;
    const lineTokens = estimateTokens(line);
    if (currentTokens + lineTokens > maxTokens && current.length > 0) {
      chunks.push({ text: current.join('\n'), lineStart, lineEnd: i });
      // Overlap: keep last N lines
      const overlapLines = Math.min(overlap, current.length);
      current = current.slice(-overlapLines);
      currentTokens = current.reduce((sum, l) => sum + estimateTokens(l), 0);
      lineStart = i - overlapLines + 1;
    }
    current.push(line);
    currentTokens += lineTokens;
  }

  if (current.length > 0) {
    chunks.push({ text: current.join('\n'), lineStart, lineEnd: lines.length });
  }

  return chunks;
}

function collectFiles(rootDir: string, dir: string = ''): string[] {
  const fullPath = path.join(rootDir, dir);
  const files: string[] = [];

  try {
    const entries = readdirSync(fullPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith('.') && entry.name !== '.context-config.json') continue;
      if (IGNORED_DIRS.has(entry.name)) continue;

      const relativePath = dir ? `${dir}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        files.push(...collectFiles(rootDir, relativePath));
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (CODE_EXTENSIONS.has(ext) || DOC_EXTENSIONS.has(ext) || SCHEMA_EXTENSIONS.has(ext)) {
          files.push(relativePath);
        }
      }
    }
  } catch { /* skip unreadable dirs */ }

  return files;
}

/** Build the context index from source files. */
export async function buildIndex(options: BuildIndexOptions = {}): Promise<BuildIndexResult> {
  const startTime = performance.now();
  const rootDir = options.rootDir ?? process.cwd();
  const config = loadConfig(rootDir);
  const indexPath = resolveIndexPath(rootDir, config);
  const warnings: string[] = [];

  mkdirSync(indexPath, { recursive: true });

  // Collect files
  const files = collectFiles(rootDir);
  const provider = createEmbeddingProvider(config);
  const records: IndexRecord[] = [];
  const countsByKind: Record<string, number> = {};

  // Chunk and create records
  for (const filePath of files) {
    try {
      const fullPath = path.join(rootDir, filePath);
      const content = readFileSync(fullPath, 'utf8');
      const kind = classifyFile(filePath);
      const chunks = chunkText(content, config.max_chunk_tokens, config.sliding_window_overlap);

      for (const chunk of chunks) {
        if (estimateTokens(chunk.text) < config.min_chunk_tokens) continue;

        const chunkId = createHash('sha256')
          .update(`${filePath}:${chunk.lineStart}:${chunk.text.slice(0, 100)}`)
          .digest('hex')
          .slice(0, 16);

        const record: IndexRecord = {
          text: chunk.text,
          vector: [], // filled after embedding
          file_path: filePath,
          chunk_id: chunkId,
          kind,
          package_name: filePath.split('/')[0] ?? '',
          symbols: '',
          heading: '',
          community_id: 0,
          line_start: chunk.lineStart,
          line_end: chunk.lineEnd,
          last_modified: null,
          recency_bucket: 2,
          token_estimate: estimateTokens(chunk.text),
          search_text: chunk.text,
          concept_text: '',
        };

        records.push(record);
        countsByKind[kind] = (countsByKind[kind] ?? 0) + 1;
      }
    } catch (err) {
      warnings.push(`Failed to process ${filePath}: ${err}`);
    }
  }

  // Embed all records in batches
  const batchSize = 50;
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    const texts = batch.map((r) => r.search_text);
    try {
      const vectors = await provider.embed(texts, { inputType: 'passage' });
      for (let j = 0; j < batch.length; j += 1) {
        batch[j]!.vector = vectors[j] ?? [];
      }
    } catch (err) {
      warnings.push(`Embedding batch ${i / batchSize} failed: ${err}`);
    }
  }

  // Compute document frequency for BM25
  const docFreq: Record<string, number> = {};
  let totalLength = 0;
  for (const record of records) {
    const tokens = new Set(tokenize(record.search_text));
    totalLength += tokens.size;
    for (const token of tokens) {
      docFreq[token] = (docFreq[token] ?? 0) + 1;
    }
  }

  // Serialize to JSONL
  const recordsPath = path.join(indexPath, 'records.jsonl');
  const jsonl = records.map((r) => JSON.stringify(r)).join('\n');
  writeFileSync(recordsPath, jsonl, 'utf8');

  // Write manifest
  const manifest = {
    version: 2,
    backend: 'json',
    built_at: new Date().toISOString(),
    git_sha: 'standalone',
    record_count: records.length,
    embedding_provider: provider.name,
    embedding_model: config.openai_model,
    embedding_dimensions: provider.dimensions,
    average_lexical_length: records.length > 0 ? totalLength / records.length : 1,
    document_frequency: docFreq,
  };
  writeFileSync(path.join(indexPath, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');

  return {
    indexFile: recordsPath,
    recordCount: records.length,
    countsByKind,
    warnings,
    durationMs: Math.round(performance.now() - startTime),
  };
}
