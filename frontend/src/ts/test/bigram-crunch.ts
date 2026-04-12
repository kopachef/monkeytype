import { randomElementFromArray } from "../utils/arrays";
import { Wordset } from "./wordset";

const bigramScores: { [bigram: string]: BigramScore } = {};
const wordSampleSize = 40;
const wordFreqMultiplier = 1.8;
const maxWeightedWords = 300;

class BigramScore {
  public score: number;
  public misses: number;
  public occurrences: number;

  constructor() {
    this.score = 0.0;
    this.misses = 0;
    // Avoid log(0). Occurrences are updated during word selection while misses
    // are updated during typing, so the score needs an initial denominator.
    this.occurrences = 2;
  }

  // Update scoring to consider both the count of failures and the log of occurrences
  updateScore(): void {
    if (this.occurrences > 0 && this.misses > 0) {
      // Ensure we don't divide by zero or take log of zero
      this.score =
        (this.misses / this.occurrences) * Math.log(this.occurrences);
    }
  }

  incrementMisses(): void {
    this.misses++;
    this.updateScore();
  }

  incrementOccurrences(): void {
    this.occurrences++;
    this.updateScore();
  }
}

export function updateBigramScore(
  currentWord: string,
  currentInput: string | null,
  isCorrect: boolean
): void {
  if (!isCorrect) {
    const bigramStartIndex = (currentInput?.length ?? 0) - 1;
    const bigram = currentWord.slice(bigramStartIndex, bigramStartIndex + 2);
    if (bigram.length < 2) {
      // Sometimes single letter characters may be found reject them here.
      return;
    }
    if (!(bigram in bigramScores)) {
      bigramScores[bigram] = new BigramScore();
    }
    bigramScores[bigram]?.incrementMisses();
  }
}

export function getWord(wordset: Wordset): string {
  // Identify weak bigrams with the highest scores
  const weakBigrams = Object.keys(bigramScores)
    .sort(
      (a, b) => (bigramScores[b]?.score ?? 0) - (bigramScores[a]?.score ?? 0)
    )
    .slice(0, 10);

  // Select words that contain any of the weak bigrams
  let filteredWords = wordset.words.filter((word) =>
    weakBigrams.some((bigram) => word.includes(bigram))
  );

  // If the filtered list is too short, add more words
  if (filteredWords.length < wordSampleSize) {
    const additionalWords = wordset.words.filter(
      (word) => !filteredWords.includes(word)
    );
    filteredWords = filteredWords.concat(
      additionalWords.slice(0, wordSampleSize - filteredWords.length)
    );
  }

  // Multiply the frequency of each word based on its bigram score
  const weightedWords: string[] = [];
  let currentCount = 0;

  filteredWords.forEach((word) => {
    let weight = wordFreqMultiplier;
    weakBigrams.forEach((bigram) => {
      if (word.includes(bigram) && bigramScores[bigram]) {
        weight *= Math.max(bigramScores[bigram]?.score ?? 0, 1); // Ensure weight is at least 1
      }
    });
    const additions = Math.ceil(weight);

    // Limit the number of additions based on the remaining space in weightedWords
    const allowedAdditions = Math.min(
      additions,
      maxWeightedWords - currentCount
    );

    for (let i = 0; i < allowedAdditions; i++) {
      weightedWords.push(word);
      currentCount++;

      // Stop if we reach the maximum limit
      if (currentCount >= maxWeightedWords) break;
    }
  });

  let chosenWord = randomElementFromArray(weightedWords);

  // Increment occurrences for each bigram in the chosen word if we are tracking
  if (chosenWord) {
    for (let i = 0; i < chosenWord.length - 1; i++) {
      const bigram = chosenWord.substring(i, i + 2);
      if (bigramScores[bigram]) {
        bigramScores[bigram]?.incrementOccurrences();
        bigramScores[bigram]?.updateScore();
      }
    }
  } else {
    chosenWord = randomElementFromArray(wordset.words);
  }
  return chosenWord;
}

// Debugging purposes.
export function logBigramScores(): boolean {
  const bigramScoresForLogging: {
    [key: string]: { score: number; occurrences: number; misses: number };
  } = {};

  Object.keys(bigramScores).forEach((bigram) => {
    const score = bigramScores[bigram];
    if (score) {
      bigramScoresForLogging[bigram] = {
        score: score.score,
        misses: score.misses,
        occurrences: score.occurrences,
      };
    }
  });

  console.debug(
    "Current Bigram Scores Table:",
    JSON.stringify(bigramScoresForLogging, null, 2)
  );
  return true;
}
