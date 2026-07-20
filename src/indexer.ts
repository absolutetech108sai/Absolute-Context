// ─── Indexer ─────────────────────────────────────────────────────────────────
// Builds the context index from source files. Walks the repo, chunks files,
// embeds chunks, and serializes to JSONL + manifest.

import { createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
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

function collectFiles(
  rootDir: string,
  dir: string = '',
  extraIgnored: Set<string> = new Set(),
): string[] {
  const fullPath = path.join(rootDir, dir);
  const files: string[] = [];

  try {
    const entries = readdirSync(fullPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith('.') && entry.name !== '.context-config.json') continue;
      if (IGNORED_DIRS.has(entry.name)) continue;
      if (extraIgnored.has(entry.name)) continue;

      const relativePath = dir ? `${dir}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        files.push(...collectFiles(rootDir, relativePath, extraIgnored));
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

/** Skip files larger than this (generated/minified/lock artifacts). */
const MAX_FILE_BYTES = 1024 * 1024; // 1 MB

/**
 * Build the context index from source files.
 *
 * Memory-efficient streaming design: processes ONE file at a time —
 * chunk → embed → write to JSONL immediately — so peak memory is a single
 * file's chunks plus the token→docCount map, NOT the whole corpus. This is
 * what lets the engine index large monorepos without exhausting the heap.
 */
export async function buildIndex(options: BuildIndexOptions = {}): Promise<BuildIndexResult> {
  const startTime = performance.now();
  const rootDir = options.rootDir ?? process.cwd();
  const config = loadConfig(rootDir);
  const indexPath = resolveIndexPath(rootDir, config);
  const warnings: string[] = [];

  mkdirSync(indexPath, { recursive: true });

  // Scope control: optionally restrict to specific top-level dirs and/or
  // skip extra dirs. A code engine pointed at a monorepo full of doc dumps
  // should index source, not 285K paragraphs of audit prose.
  const extraIgnored = new Set(options.excludeDirs ?? []);
  let files: string[];
  if (options.includeDirs && options.includeDirs.length > 0) {
    files = [];
    for (const top of options.includeDirs) {
      files.push(...collectFiles(rootDir, top, extraIgnored));
    }
  } else {
    files = collectFiles(rootDir, '', extraIgnored);
  }
  const maxRecords = options.maxRecords ?? Infinity;

  const provider = createEmbeddingProvider(config);
  const countsByKind: Record<string, number> = {};
  const docFreq = new Map<string, number>(); // token → number of docs containing it
  let totalLength = 0;
  let recordCount = 0;

  const recordsPath = path.join(indexPath, 'records.jsonl');
  const out = createWriteStream(recordsPath, 'utf8');

  for (const filePath of files) {
    if (recordCount >= maxRecords) {
      warnings.push(`Stopped at maxRecords=${maxRecords}`);
      break;
    }
    try {
      const fullPath = path.join(rootDir, filePath);
      const stat = statSync(fullPath);
      if (stat.size > MAX_FILE_BYTES) {
        warnings.push(`Skipped (>${MAX_FILE_BYTES / 1024}KB): ${filePath}`);
        continue;
      }

      const content = readFileSync(fullPath, 'utf8');
      const kind = classifyFile(filePath);
      const chunks = chunkText(content, config.max_chunk_tokens, config.sliding_window_overlap);

      // Build this file's records (held in memory only until written)
      const fileRecords: IndexRecord[] = [];
      for (const chunk of chunks) {
        if (estimateTokens(chunk.text) < config.min_chunk_tokens) continue;
        const chunkId = createHash('sha256')
          .update(`${filePath}:${chunk.lineStart}:${chunk.text.slice(0, 100)}`)
          .digest('hex')
          .slice(0, 16);
        fileRecords.push({
          text: chunk.text,
          vector: [],
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
        });
      }
      if (fileRecords.length === 0) continue;

      // Embed this file's chunks (batch)
      try {
        const vectors = await provider.embed(
          fileRecords.map((r) => r.search_text),
          { inputType: 'passage' },
        );
        for (let j = 0; j < fileRecords.length; j += 1) {
          fileRecords[j]!.vector = vectors[j] ?? [];
        }
      } catch (err) {
        warnings.push(`Embedding failed for ${filePath}: ${err}`);
      }

      // Update doc-frequency stats and stream records to disk immediately
      for (const record of fileRecords) {
        const uniqueTokens = new Set(tokenize(record.search_text));
        totalLength += uniqueTokens.size;
        for (const token of uniqueTokens) {
          docFreq.set(token, (docFreq.get(token) ?? 0) + 1);
        }
        out.write(`${JSON.stringify(record)}\n`);
        recordCount += 1;
        countsByKind[kind] = (countsByKind[kind] ?? 0) + 1;
      }
    } catch (err) {
      warnings.push(`Failed to process ${filePath}: ${err}`);
    }
  }

  // Flush the write stream
  out.end();
  await new Promise<void>((resolve, reject) => {
    out.on('finish', resolve);
    out.on('error', reject);
  });

  // Write manifest (docFreq as a plain object)
  const manifest = {
    version: 2,
    backend: 'json',
    built_at: new Date().toISOString(),
    git_sha: 'standalone',
    record_count: recordCount,
    embedding_provider: provider.name,
    embedding_model: config.openai_model,
    embedding_dimensions: provider.dimensions,
    average_lexical_length: recordCount > 0 ? totalLength / recordCount : 1,
    document_frequency: Object.fromEntries(docFreq),
  };
  writeFileSync(path.join(indexPath, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');

  return {
    indexFile: recordsPath,
    recordCount,
    countsByKind,
    warnings,
    durationMs: Math.round(performance.now() - startTime),
  };
}
