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

## Discussion

### What we found
Semantic surprisal positively predicts both word duration (r=0.12 after controls, p<0.001)
and pre-word silence (r=0.08, p<0.001), consistent across two speakers and two speech genres.

### Problems addressed
- **Word length confound:** surprising words tend to be longer. Addressed with partial
  correlation controlling for word length — effect persists but shrinks (r=0.33 → r=0.12),
  confirming a genuine surprisal effect above and beyond physical word length.
- **Context length artifact:** distance-from-mean embeddings were inflated by sentence
  position. Addressed by switching to surprisal, which uses GPT-2's prediction mechanism
  directly and is position-independent.
- **Noisy pre-gap measurements:** WhisperX alignment isn't perfect at word boundaries,
  adding noise to pre-gap estimates.

### Limitations
- Only two highly practiced public speakers — may not generalize to spontaneous conversation
- Syllable count and other variables may be a better control than character length
- Surprisal does not equal semantic richness exactly — unexpected words aren't always semantically rich

### Next steps
- Control for syllable count instead of character length
- Layer comparison: does surprisal from layer 1 vs 18 vs 36 predict timing differently?
- Apply to more speakers and spontaneous speech genres

### Connection to Hayden Lab research
This project is a behavioral proof of concept for the neural analysis at Hayden Lab.
The core GPT-2 Large pipeline built here, transcription, word-level alignment, and
layer 36 embedding extraction, is the same infrastructure used in Franch et al. (2026)
to predict hippocampal single-neuron firing rates during naturalistic speech. The real
project extends that work by adding social gaze as a modulating variable, asking whether
hippocampal semantic encoding is stronger during moments of social visual attention,
connecting to broader questions about how the brain binds social cues and semantic content
during naturalistic human interaction.
