// ─── Tokenizer ───────────────────────────────────────────────────────────────
// Text tokenization, identifier squashing, bigrams, and token counting.

const TOKEN_REGEX = /[a-z0-9_]+/g;

/** Normalize text for tokenization: lowercase, strip non-alphanumeric. */
export function normalizeForTokens(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9_\s]/g, ' ');
}

/** Split text into lowercase alphanumeric tokens. */
export function tokenize(text: string): string[] {
  const normalized = normalizeForTokens(text);
  return normalized.match(TOKEN_REGEX) ?? [];
}

/** Squash an identifier to lowercase alphanumeric only (for fuzzy matching). */
export function squashIdentifier(input: string): string {
  return input.toLowerCase().replace(/[^a-z0-9]/g, '');
}

/** Generate concept token variants (singular/plural, common suffixes). */
export function conceptTokenVariants(token: string): string[] {
  const variants = [token];
  if (token.endsWith('s') && token.length > 3) {
    variants.push(token.slice(0, -1));
  } else if (token.length > 2) {
    variants.push(`${token}s`);
  }
  if (token.endsWith('ing') && token.length > 5) {
    variants.push(token.slice(0, -3));
    variants.push(token.slice(0, -3) + 'e');
  }
  if (token.endsWith('ed') && token.length > 4) {
    variants.push(token.slice(0, -2));
    variants.push(token.slice(0, -2) + 'e');
  }
  if (token.endsWith('tion') && token.length > 5) {
    variants.push(token.slice(0, -4) + 'te');
  }
  return variants;
}

/** Build a frequency map from a token array. */
export function createTokenCountMap(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  return counts;
}

/** Estimate token count from text length (~4 chars per token). */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/** Check if needle tokens appear in order within haystack tokens. */
export function containsOrderedTokenSequence(
  haystackTokens: string[],
  needleTokens: string[],
): boolean {
  if (needleTokens.length === 0 || needleTokens.length > haystackTokens.length) {
    return false;
  }
  for (let start = 0; start <= haystackTokens.length - needleTokens.length; start += 1) {
    let matched = true;
    for (let offset = 0; offset < needleTokens.length; offset += 1) {
      if (haystackTokens[start + offset] !== needleTokens[offset]) {
        matched = false;
        break;
      }
    }
    if (matched) return true;
  }
  return false;
}

/** Generate bigram pairs from a token array. */
export function bigrams(tokens: string[]): string[] {
  const pairs: string[] = [];
  for (let index = 0; index < tokens.length - 1; index += 1) {
    pairs.push(`${tokens[index]} ${tokens[index + 1]}`);
  }
  return pairs;
}
