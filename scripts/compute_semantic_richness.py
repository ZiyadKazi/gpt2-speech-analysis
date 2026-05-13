"""
Script: compute_semantic_richness.py
=====================================
Purpose: Compute semantic richness (surprisal) for each word using GPT-2 Large.
         Surprisal = how unexpected was this word given its context.
         High surprisal = semantically rich and specific.
         Low surprisal = predictable and generic.

Usage:
    python scripts/compute_semantic_richness.py \
        --transcript data/processed/brenebrown_5min_transcript.json \
        --word_data outputs/embeddings/brenebrown_5min_layer36_word_data.csv

Input:  data/processed/<name>_transcript.json
        outputs/embeddings/<name>_layer<n>_word_data.csv
Output: outputs/embeddings/<name>_layer<n>_richness.csv

Author: Ziyad Kazi | Hayden Lab | Summer 2026
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


# ── Argument Parsing ───────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute semantic richness (surprisal) from GPT-2 Large"
    )
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to transcript JSON file (from transcribe.py)"
    )
    parser.add_argument(
        "--word_data",
        type=str,
        required=True,
        help="Path to word data CSV file (from extract_embeddings.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/embeddings",
        help="Directory to save richness CSV (default: outputs/embeddings)"
    )
    return parser.parse_args()


# ── Split into sentences ───────────────────────────────────────────────────
def split_into_sentences(words):
    """
    Group words into sentences using punctuation as boundaries.
    Each sentence resets the GPT-2 context window.
    Returns list of sentences, each a list of word dicts.
    """
    sentences = []
    current = []

    for word in words:
        current.append(word)
        if any(word["word"].endswith(p) for p in [".", "!", "?"]):
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    print(f"Split into {len(sentences)} sentences")
    return sentences


# ── Surprisal ──────────────────────────────────────────────────────────────
def compute_surprisal(sentences, tokenizer, model):
    """
    Compute surprisal for each word using GPT-2 Large.
    Surprisal = -log P(word | context) measured in nats.
    High surprisal = unexpected, semantically specific word.
    Low surprisal = predictable, generic word.

    Args:
        sentences: list of sentences, each a list of word dicts
        tokenizer: GPT2Tokenizer
        model: GPT2LMHeadModel

    Returns:
        surprisals: np.array of shape (n_words,)
        surprisals_normalized: surprisals scaled to [0, 1]
    """
    loss_fn = CrossEntropyLoss(reduction="none")
    all_surprisals = []

    for sentence in tqdm(sentences, desc="Computing surprisal"):
        words_in_sentence = [w["word"].strip() for w in sentence]

        for word_idx in range(len(words_in_sentence)):

            # First word has no context — assign None, fill with mean later
            if word_idx == 0:
                all_surprisals.append(None)
                continue

            # Build context and target
            context = " ".join(words_in_sentence[:word_idx])
            target  = " " + words_in_sentence[word_idx]

            # Tokenize separately to know boundary
            context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
            target_ids  = tokenizer(target,  return_tensors="pt")["input_ids"]

            # Concatenate into full sequence
            full_ids = torch.cat([context_ids, target_ids], dim=1)
            n_context = context_ids.shape[1]

            # Run model
            with torch.no_grad():
                outputs = model(full_ids, labels=full_ids)

            # Extract logits at positions predicting target tokens
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)
            target_logits = logits[0, n_context-1:-1, :]  # predictions for target
            target_labels = target_ids[0]                  # actual target tokens

            # Compute loss (= negative log probability = surprisal)
            token_losses = loss_fn(target_logits, target_labels)

            # Average across sub-tokens → one surprisal value per word
            surprisal = token_losses.mean().item()
            all_surprisals.append(surprisal)

    # Replace None (first words) with mean surprisal
    mean_surprisal = np.mean([s for s in all_surprisals if s is not None])
    all_surprisals = [s if s is not None else mean_surprisal
                      for s in all_surprisals]

    surprisals = np.array(all_surprisals)
    surprisals_normalized = ((surprisals - surprisals.min()) /
                             (surprisals.max() - surprisals.min()))

    return surprisals, surprisals_normalized


# ── Pre-gap ────────────────────────────────────────────────────────────────
def compute_pre_gap(df):
    """
    Compute the silence before each word starts.
    pre_gap = start time of this word - end time of previous word.
    First word of each sentence gets pre_gap = 0.

    Args:
        df: DataFrame with 'start', 'end', 'sentence_idx' columns

    Returns:
        pre_gap: np.array of shape (n_words,)
    """
    pre_gap = np.zeros(len(df))

    for i in range(1, len(df)):
        same_sentence = (df.iloc[i]["sentence_idx"] ==
                         df.iloc[i-1]["sentence_idx"])
        if same_sentence:
            gap = df.iloc[i]["start"] - df.iloc[i-1]["end"]
            pre_gap[i] = max(0, gap)  # clamp negatives to 0

    return pre_gap


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    transcript_path = Path(args.transcript)
    word_data_path  = Path(args.word_data)
    output_dir      = Path(args.output_dir)

    # Build output filename from word_data filename
    stem        = word_data_path.stem.replace("_word_data", "")
    output_path = output_dir / f"{stem}_richness.csv"

    # ── Checks ─────────────────────────────────────────────────────────────
    assert transcript_path.exists(), f"Transcript not found: {transcript_path}"
    assert word_data_path.exists(),  f"Word data not found: {word_data_path}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transcript:  {transcript_path}")
    print(f"Word data:   {word_data_path}")
    print(f"Output:      {output_path}")

    # ── Step 1: Load data ──────────────────────────────────────────────────
    print("\nStep 1: Loading data...")
    df = pd.read_csv(word_data_path)
    print(f"Words loaded: {len(df)}")

    with open(transcript_path) as f:
        transcript_data = json.load(f)
    words     = transcript_data["word_segments"]
    sentences = split_into_sentences(words)
    print(f"Total words in transcript: {len(words)}")

    # ── Step 2: Load GPT-2 Large ───────────────────────────────────────────
    print("\nStep 2: Loading GPT-2 Large...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model     = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model.eval()
    print("Model loaded")

    # ── Step 3: Compute surprisal ──────────────────────────────────────────
    print("\nStep 3: Computing surprisal (this takes a few minutes)...")
    richness, richness_normalized = compute_surprisal(sentences, tokenizer, model)
    print(f"Surprisal — min: {richness.min():.4f}, "
          f"max: {richness.max():.4f}, "
          f"mean: {richness.mean():.4f}")

    # ── Step 4: Compute pre-gap ────────────────────────────────────────────
    print("\nStep 4: Computing pre-word pause (pre_gap)...")
    pre_gap = compute_pre_gap(df)
    print(f"Pre-gap — min: {pre_gap.min():.4f}, "
          f"max: {pre_gap.max():.4f}, "
          f"mean: {pre_gap.mean():.4f}")

    # ── Step 5: Build output dataframe ────────────────────────────────────
    print("\nStep 5: Building output dataframe...")
    df_out = df.copy()
    df_out["richness"]            = richness
    df_out["richness_normalized"] = richness_normalized
    df_out["pre_gap"]             = pre_gap
    df_out["word_length"]         = df_out["word"].str.len()

    # ── Step 6: Save ───────────────────────────────────────────────────────
    print("\nStep 6: Saving...")
    df_out.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    # ── Step 7: Sanity check ───────────────────────────────────────────────
    print("\nSanity check — top 10 most surprising words:")
    print(f"{'Word':<20} {'Surprisal':>10} {'Normalized':>12} "
          f"{'Duration':>10} {'Pre-gap':>10}")
    print("-" * 66)
    for _, row in df_out.nlargest(10, "richness").iterrows():
        print(f"{row['word']:<20} {row['richness']:>10.4f} "
              f"{row['richness_normalized']:>12.4f} "
              f"{row['duration']:>10.4f} "
              f"{row['pre_gap']:>10.4f}")

    print("\nSanity check — top 10 least surprising words:")
    print(f"{'Word':<20} {'Surprisal':>10} {'Normalized':>12} "
          f"{'Duration':>10} {'Pre-gap':>10}")
    print("-" * 66)
    for _, row in df_out.nsmallest(10, "richness").iterrows():
        print(f"{row['word']:<20} {row['richness']:>10.4f} "
              f"{row['richness_normalized']:>12.4f} "
              f"{row['duration']:>10.4f} "
              f"{row['pre_gap']:>10.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()