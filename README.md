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
- TED Talk: Brené Brown "The Power of Vulnerability" (first 5 minutes)

## Author
Ziyad Kazi
Hayden Lab, Baylor College of Medicine
Summer 2026
