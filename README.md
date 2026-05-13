# GPT-2 Speech Semantic Analysis

## Project
Investigating whether semantic richness of speech (measured via GPT-2 Large 
embeddings) predicts speech rate in naturalistic TED Talk speech.

## Scientific Question
Do speakers slow down during semantically rich moments?

## Pipeline
1. Parse transcript with word-level timestamps (WhisperX)
2. Extract GPT-2 Large embeddings per word (layer 37)
3. Reduce dimensions with PCA (1280 -> 50)
4. Compute semantic richness (distance from mean embedding)
5. Compute speech rate per time window
6. Correlate semantic richness with speech rate
7. Visualize results

## Data
Two naturalistic speech samples from public figures:

**Speaker 1: Brené Brown**
Talk: "The Power of Vulnerability" — TED Talk (first 5 minutes)
Source: https://www.youtube.com/watch?v=iCvmsMzlF7o
Words: 878

**Speaker 2: Ben Horowitz**
Talk: "Don't Follow Your Passion" — Columbia Commencement Address (minutes 1-9)
Source: https://www.youtube.com/watch?v=uaSqh4DiQSw
Words: 1386

Total words analyzed: 2,264 across two speakers, two genres (TED Talk vs commencement),
two content domains (vulnerability/social science vs entrepreneurship/culture).

## Extensions
- **Layer comparison analysis**: Extract embeddings from layers 1, 8, 16, 24, 32, 37
  and compare how well each layer's semantic structure correlates with speech rate.
  This tests the hypothesis that upper-middle layers capture richer semantics than
  the final layer for downstream prediction tasks.
