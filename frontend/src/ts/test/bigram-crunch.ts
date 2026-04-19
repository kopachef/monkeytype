import { z } from "zod";

import { Config } from "../config/store";
import { LocalStorageWithSchema } from "../utils/local-storage-with-schema";
import { isDevEnvironment } from "../utils/env";
import * as TestInput from "./test-input";
import { Wordset } from "./wordset";

const topCandidateCount = 5;
const maxStoredBigrams = 300;
const supportSampleSize = 500;
const minSupportedWordRatio = 0.5;
const wordIndexSampleSize = 5000;
const topScoredBigramCount = 10;
const maxCandidatePool = 120;
const recentWordLimit = 25;
const movingAverageWindow = 50;
const saveEveryUpdates = 10;
// Weakspot already targets weak single characters by averaging key spacing with
// an error penalty. Bigram Crunch keeps that targeted-practice idea, but scores
// adjacent character pairs so slow or mistyped transitions can be practised
// directly.
// These tuning values came from a small simulation harness:
// https://github.com/kopachef/bigram-crunch-lab
// Longer write-up: https://martinnn.com/blog/bigram-crunch/
const scoringConfig = {
  confidenceAttempts: 10,
  timingBaselineMs: 180,
  minTimingMs: 20,
  // Treat timings beyond this as pauses/outliers and skip the whole bigram
  // update so distracted breaks do not count as attempts or misses.
  maxTimingMs: 1200,
  missRateWeight: 0.65,
  timingWeight: 0.35,
  explorationRate: 0.05,
} as const;
const debugLogging = isDevEnvironment();

// Bigram Crunch intentionally starts with Latin alphabetic pairs only. Other
// scripts need separate segmentation rules, so they are ignored rather than
// folded into misleading two-code-unit scores.
const supportedBigramRegex = /^[a-z]{2}$/;

const BigramStatsSchema = z
  .object({
    attempts: z.number().int().nonnegative(),
    misses: z.number().int().nonnegative(),
    averageMs: z.number().nonnegative(),
    score: z.number().nonnegative(),
    lastSeen: z.number().int().nonnegative(),
  })
  .strict();

const BigramCrunchStorageSchema = z
  .object({
    version: z.literal(1),
    bigrams: z.record(
      z.string().regex(supportedBigramRegex),
      BigramStatsSchema,
    ),
  })
  .strict();

type BigramStats = z.infer<typeof BigramStatsSchema>;
type BigramCrunchStorage = z.infer<typeof BigramCrunchStorageSchema>;

type RankedBigram = BigramStats & {
  bigram: string;
};

type CacheStatus = "fresh" | "resumed";
type SupportStatus = {
  supported: boolean;
  reason: string | undefined;
  checkedWords: number;
  supportedWords: number;
  supportedWordRatio: number;
  supportedBigramCount: number;
  unsupportedExamples: string[];
};
type WordCandidateIndex = {
  wordset: Wordset;
  bigramToWords: Map<string, string[]>;
};
type BigramSpacing = {
  spacing: number | undefined;
  shouldSkip: boolean;
};

const storage = new LocalStorageWithSchema<BigramCrunchStorage>({
  key: "bigramCrunchStats",
  schema: BigramCrunchStorageSchema,
  fallback: createEmptyStorage(),
});

let statsCache: BigramCrunchStorage | undefined;
let unsavedUpdates = 0;
let cacheStatus: CacheStatus | undefined;
let supportStatus: SupportStatus | undefined;
let supportStatusWordset: Wordset | undefined;
let wordCandidateIndex: WordCandidateIndex | undefined;
const recentWords: string[] = [];

function createEmptyStorage(): BigramCrunchStorage {
  return {
    version: 1,
    bigrams: {},
  };
}

function getStats(): BigramCrunchStorage {
  if (statsCache === undefined) {
    statsCache = storage.get();
    cacheStatus =
      Object.keys(statsCache.bigrams).length === 0 ? "fresh" : "resumed";
  }

  return statsCache;
}

function persistStats(force = false): void {
  if (statsCache === undefined || unsavedUpdates === 0) return;
  if (!force && unsavedUpdates < saveEveryUpdates) return;

  pruneStats(statsCache);
  storage.set(statsCache);
  unsavedUpdates = 0;
}

function pruneStats(stats: BigramCrunchStorage): void {
  const entries = Object.entries(stats.bigrams);
  if (entries.length <= maxStoredBigrams) return;

  stats.bigrams = Object.fromEntries(
    entries
      .sort(([, a], [, b]) => b.score - a.score || b.lastSeen - a.lastSeen)
      .slice(0, maxStoredBigrams),
  ) as BigramCrunchStorage["bigrams"];
}

function createStats(): BigramStats {
  return {
    attempts: 0,
    misses: 0,
    averageMs: 0,
    score: 0,
    lastSeen: Date.now(),
  };
}

function normalizeBigram(bigram: string): string | undefined {
  const normalized = bigram.toLowerCase();
  if (!supportedBigramRegex.test(normalized)) return undefined;
  return normalized;
}

function getTargetBigram(
  currentWord: string,
  currentInput: string | null,
): string | undefined {
  const index = currentInput?.length ?? 0;
  if (index <= 0) return undefined;

  return normalizeBigram(currentWord.slice(index - 1, index + 1));
}

function isAlignedWithTargetPrefix(
  currentWord: string,
  currentInput: string | null,
): boolean {
  if (currentInput === null) return false;

  return currentInput === currentWord.slice(0, currentInput.length);
}

function getWordBigrams(word: string): string[] {
  const bigrams: string[] = [];
  for (let i = 0; i < word.length - 1; i++) {
    const bigram = normalizeBigram(word.slice(i, i + 2));
    if (bigram !== undefined) {
      bigrams.push(bigram);
    }
  }
  return bigrams;
}

function getSupportStatus(wordset: Wordset): SupportStatus {
  if (supportStatusWordset === wordset && supportStatus !== undefined) {
    return supportStatus;
  }

  const words = wordset.words.slice(0, supportSampleSize);
  const supportedBigrams = new Set<string>();
  let supportedWords = 0;
  const unsupportedExamples: string[] = [];

  for (const word of words) {
    const bigrams = getWordBigrams(word);
    if (bigrams.length > 0) {
      supportedWords++;
      bigrams.forEach((bigram) => supportedBigrams.add(bigram));
    } else if (unsupportedExamples.length < 5) {
      unsupportedExamples.push(word);
    }
  }

  const supportedWordRatio =
    words.length === 0 ? 0 : supportedWords / words.length;
  const supported = supportedWordRatio >= minSupportedWordRatio;

  supportStatus = {
    supported,
    reason: supported
      ? undefined
      : `Only ${(supportedWordRatio * 100).toFixed(1)}% of sampled words contain adjacent a-z letter pairs.`,
    checkedWords: words.length,
    supportedWords,
    supportedWordRatio,
    supportedBigramCount: supportedBigrams.size,
    unsupportedExamples,
  };
  supportStatusWordset = wordset;

  return supportStatus;
}

function getWordCandidateIndex(wordset: Wordset): WordCandidateIndex {
  if (wordCandidateIndex?.wordset === wordset) {
    return wordCandidateIndex;
  }

  const bigramToWords = new Map<string, string[]>();
  // Build a bounded index so large word lists do not pay a full scan cost while
  // still giving word selection direct access to known weak bigrams.
  for (const word of wordset.words.slice(0, wordIndexSampleSize)) {
    const uniqueBigrams = new Set(getWordBigrams(word));
    for (const bigram of uniqueBigrams) {
      const words = bigramToWords.get(bigram) ?? [];
      words.push(word);
      bigramToWords.set(bigram, words);
    }
  }

  wordCandidateIndex = {
    wordset,
    bigramToWords,
  };
  recentWords.length = 0;

  return wordCandidateIndex;
}

function getSpacingForBigram(currentInputLength: number): BigramSpacing {
  const timings = TestInput.keypressTimings.spacing.array;
  if (timings.length === 0 || typeof timings === "string") {
    return { spacing: undefined, shouldSkip: false };
  }

  const timingIndex = currentInputLength - 1;
  const spacing = timings[timingIndex];
  if (typeof spacing !== "number" || !Number.isFinite(spacing)) {
    return { spacing: undefined, shouldSkip: false };
  }

  if (
    spacing < scoringConfig.minTimingMs ||
    spacing > scoringConfig.maxTimingMs
  ) {
    return { spacing: undefined, shouldSkip: true };
  }

  return { spacing, shouldSkip: false };
}

function updateMovingAverage(
  currentAverage: number,
  nextValue: number,
  previousAttempts: number,
): number {
  if (previousAttempts === 0 || currentAverage === 0) {
    return nextValue;
  }

  const count = Math.min(previousAttempts + 1, movingAverageWindow);
  const adjustRate = 1 / count;
  return nextValue * adjustRate + currentAverage * (1 - adjustRate);
}

function calculateScore(stats: BigramStats): number {
  if (stats.attempts === 0) return 0;

  const missRate = stats.misses / stats.attempts;
  // A few early mistakes should not dominate forever; confidence lets the score
  // ramp up as the bigram gets enough attempts to be meaningful.
  const timingPenalty = Math.min(
    Math.max(0, stats.averageMs - scoringConfig.timingBaselineMs) /
      scoringConfig.timingBaselineMs,
    2,
  );
  const confidence = Math.min(
    1,
    stats.attempts / scoringConfig.confidenceAttempts,
  );

  return (
    confidence *
    (missRate * scoringConfig.missRateWeight +
      timingPenalty * scoringConfig.timingWeight)
  );
}

function scoreWord(word: string): number {
  const stats = getStats();
  const bigramScores = getWordBigrams(word)
    .map((bigram) => stats.bigrams[bigram]?.score ?? 0)
    .filter((score) => score > 0)
    .sort((a, b) => b - a);

  // Limit each word to its strongest few pairs so long words do not win just
  // because they contain more bigrams.
  return bigramScores.slice(0, 3).reduce((total, score) => total + score, 0);
}

function weightedChoice(
  candidates: { word: string; score: number }[],
): string | undefined {
  const totalScore = candidates.reduce(
    (total, candidate) => total + candidate.score,
    0,
  );
  if (totalScore <= 0) return undefined;

  let target = Math.random() * totalScore;
  for (const candidate of candidates) {
    target -= candidate.score;
    if (target <= 0) {
      return candidate.word;
    }
  }

  return candidates[candidates.length - 1]?.word;
}

function rememberWord(word: string): void {
  recentWords.push(word);
  if (recentWords.length > recentWordLimit) {
    recentWords.shift();
  }
}

function chooseRandomWord(wordset: Wordset): string {
  const word = wordset.randomWord("normal");
  rememberWord(word);
  return word;
}

export function updateBigramScore(
  currentWord: string,
  currentInput: string | null,
  isCorrect: boolean,
): void {
  if (supportStatus?.supported === false) return;
  // If the input is already shifted, for example "sstatte" while targeting
  // "state", later keypresses are marked incorrect because of alignment rather
  // than because the current bigram is weak. Do not let that cascade poison the
  // bigram stats.
  if (!isAlignedWithTargetPrefix(currentWord, currentInput)) return;

  const bigram = getTargetBigram(currentWord, currentInput);
  if (bigram === undefined) return;

  const stats = getStats();
  const bigramStats = (stats.bigrams[bigram] ??= createStats());
  const previousAttempts = bigramStats.attempts;
  const { spacing, shouldSkip } = getSpacingForBigram(
    currentInput?.length ?? 0,
  );
  // Ignore the whole update after an obvious pause/outlier. Counting the
  // keypress as a normal attempt would make the stats look cleaner or worse for
  // reasons unrelated to the bigram itself.
  if (shouldSkip) return;

  bigramStats.attempts++;
  if (!isCorrect) {
    bigramStats.misses++;
  }
  if (spacing !== undefined) {
    bigramStats.averageMs = updateMovingAverage(
      bigramStats.averageMs,
      spacing,
      previousAttempts,
    );
  }
  bigramStats.score = calculateScore(bigramStats);
  bigramStats.lastSeen = Date.now();

  if (debugLogging) {
    console.log("Bigram Crunch update", {
      bigram,
      isCorrect,
      spacing,
      ...bigramStats,
    });
  }

  unsavedUpdates++;
  persistStats();
}

export function getWord(wordset: Wordset): string {
  persistStats(true);

  const currentSupport = getSupportStatus(wordset);
  if (!currentSupport.supported) {
    return chooseRandomWord(wordset);
  }

  // Keep a small amount of random exploration so the selector can still
  // discover weak bigrams it has not scored highly yet.
  if (Math.random() < scoringConfig.explorationRate) {
    return chooseRandomWord(wordset);
  }

  const index = getWordCandidateIndex(wordset);
  const candidateWords = new Set<string>();
  for (const { bigram } of getTopBigrams(topScoredBigramCount)) {
    const words = index.bigramToWords.get(bigram);
    if (words === undefined) continue;
    for (const word of words) {
      candidateWords.add(word);
      if (candidateWords.size >= maxCandidatePool) break;
    }
    if (candidateWords.size >= maxCandidatePool) break;
  }

  const candidates = [...candidateWords]
    .filter((word) => !recentWords.includes(word))
    .map((word) => ({
      word,
      score: scoreWord(word),
    }))
    .filter((candidate) => candidate.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topCandidateCount);

  const selectedWord =
    weightedChoice(candidates) ??
    weightedChoice(
      [...candidateWords]
        .map((word) => ({
          word,
          score: scoreWord(word),
        }))
        .filter((candidate) => candidate.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topCandidateCount),
    );

  if (selectedWord !== undefined) {
    rememberWord(selectedWord);
    return selectedWord;
  }

  return chooseRandomWord(wordset);
}

export function getTopBigrams(limit = 20): RankedBigram[] {
  persistStats(true);

  return getAllBigrams().slice(0, limit);
}

export function getAllBigrams(): RankedBigram[] {
  persistStats(true);

  return Object.entries(getStats().bigrams)
    .map(([bigram, stats]) => ({
      bigram,
      ...stats,
    }))
    .sort((a, b) => b.score - a.score || b.lastSeen - a.lastSeen);
}

export function getStatsForDebug(): BigramCrunchStorage {
  persistStats(true);
  return structuredClone(getStats());
}

export function getSupportStatusForDebug(): SupportStatus | undefined {
  return supportStatus === undefined
    ? undefined
    : structuredClone(supportStatus);
}

export function resetStats(): void {
  statsCache = createEmptyStorage();
  cacheStatus = "fresh";
  unsavedUpdates = 1;
  persistStats(true);
}

export function logBigramScores(): boolean {
  console.log("Current Bigram Scores Table:", getAllBigrams());
  console.table(getAllBigrams());
  return true;
}

export function logSessionStart(): boolean {
  if (!debugLogging) return true;

  const stats = getStats();
  const bigramCount = Object.keys(stats.bigrams).length;
  const currentSupportStatus = supportStatus;
  const message =
    cacheStatus === "resumed"
      ? "Bigram Crunch cache resumed"
      : "Bigram Crunch cache starting fresh";

  console.log(message, {
    language: Config.language,
    scoringConfig,
    bigramCount,
    supported: currentSupportStatus?.supported ?? "unknown",
    disabledReason: currentSupportStatus?.reason,
    supportStatus: currentSupportStatus,
    topBigrams: getTopBigrams(10),
  });

  if (currentSupportStatus?.supported === false) {
    console.log("Bigram Crunch disabled for current language", {
      language: Config.language,
      reason: currentSupportStatus.reason,
      supportStatus: currentSupportStatus,
    });
  }

  return true;
}

export function logSessionEnd(): boolean {
  if (!debugLogging) return true;

  const allBigrams = getAllBigrams();
  console.log("Bigram Crunch full cache after test", {
    bigramCount: allBigrams.length,
    bigrams: allBigrams,
  });
  console.table(allBigrams);
  return true;
}
