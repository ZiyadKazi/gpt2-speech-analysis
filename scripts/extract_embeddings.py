"""
Script: extract_embeddings.py
==============================
Purpose: Extract GPT-2 Large embeddings for each word in a transcript.
         Uses layer 37 (final transformer layer) by default, but supports
         extracting from any layer for comparison analysis.

Usage:
    python scripts/extract_embeddings.py --transcript <path_to_transcript>

Input:  data/processed/<name>_transcript.json  (from transcribe.py)
Output: outputs/embeddings/<name>_embeddings.npy   — embeddings matrix (n_words x 1280)
        outputs/embeddings/<name>_word_data.csv    — word, start, end, duration
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GPT-2 Large embeddings from a transcript"
    )
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to transcript JSON file (from transcribe.py)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=36,
        help="Which GPT-2 layer to extract (0-36, default: 36)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/embeddings",
        help="Directory to save embeddings (default: outputs/embeddings)"
    )
    return parser.parse_args()


# Load transcript
def load_transcript(transcript_path):
    """Load transcript JSON and return list of word dictionaries."""
    with open(transcript_path, "r") as f:
        data = json.load(f)

    words = data["word_segments"]
    print(f"Loaded {len(words)} words from transcript")
    return words


# Split into sentences
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
        # End sentence at punctuation
        if any(word["word"].endswith(p) for p in [".", "!", "?"]):
            sentences.append(current)
            current = []

    # Add any remaining words as final sentence
    if current:
        sentences.append(current)

    print(f"Split into {len(sentences)} sentences")
    return sentences


# Extract embeddings
def extract_embeddings(sentences, tokenizer, model, layer):
    """
    Extract GPT-2 embeddings for every word using progressive context.
    For each word, feed GPT-2 everything from sentence start up to
    and including that word.

    Returns:
        embeddings: np.array of shape (n_words, 1280)
        word_records: list of dicts with word metadata
    """
    all_embeddings = []
    all_word_records = []

    for sent_idx, sentence in enumerate(tqdm(sentences, desc="Processing sentences")):
        words_in_sentence = [w["word"].strip() for w in sentence]

        for word_idx, word_data in enumerate(sentence):
            # Build progressive context: words 0 through word_idx
            context = " ".join(words_in_sentence[:word_idx + 1])

            # Tokenize context
            inputs = tokenizer(context, return_tensors="pt")
            n_tokens = inputs["input_ids"].shape[1]

            # Run GPT-2 — extract all hidden states
            with torch.no_grad():
                outputs = model(**inputs)

            # Get hidden state from target layer
            # hidden_states[0] = embedding layer (before any transformer)
            # hidden_states[1] = after layer 1
            # hidden_states[36] = after layer 36 (final)
            hidden_states = outputs.hidden_states[layer]
            # shape: (1, n_tokens, 1280)

            # Get tokens for the target word only
            target_word = words_in_sentence[word_idx]
            target_tokens = tokenizer(
                " " + target_word, return_tensors="pt"
            )["input_ids"][0]
            n_target = len(target_tokens)

            # Average across target word's sub-tokens (last n_target tokens)
            word_embedding = hidden_states[0, -n_target:, :].mean(dim=0)
            all_embeddings.append(word_embedding.numpy())

            # Store word metadata
            all_word_records.append({
                "word":         word_data.get("word", "").strip(),
                "start":        word_data.get("start", 0),
                "end":          word_data.get("end", 0),
                "duration":     word_data.get("end", 0) - word_data.get("start", 0),
                "sentence_idx": sent_idx,
                "word_idx":     word_idx,
                "layer":        layer
            })

    embeddings = np.array(all_embeddings)  # shape: (n_words, 1280)
    return embeddings, all_word_records


# Main
def main():
    args = parse_args()

    transcript_path = Path(args.transcript)
    output_dir = Path(args.output_dir)
    stem = transcript_path.stem.replace("_transcript", "")

    # Output paths
    embeddings_path = output_dir / f"{stem}_layer{args.layer}_embeddings.npy"
    word_data_path  = output_dir / f"{stem}_layer{args.layer}_word_data.csv"

    # Checks
    assert transcript_path.exists(), f"Transcript not found: {transcript_path}"
    assert 0 <= args.layer <= 36, "Layer must be between 0 and 36"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transcript:  {transcript_path}")
    print(f"Layer:       {args.layer}")
    print(f"Embeddings:  {embeddings_path}")
    print(f"Word data:   {word_data_path}")

    # Step 1: Load transcript
    print("\nStep 1: Loading transcript...")
    words = load_transcript(transcript_path)

    # Step 2: Split into sentences
    print("\nStep 2: Splitting into sentences...")
    sentences = split_into_sentences(words)

    # Step 3: Load GPT-2 Large
    print("\nStep 3: Loading GPT-2 Large model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2Model.from_pretrained("gpt2-large", output_hidden_states=True)
    model.eval()
    print("Model loaded")
    print(f"Number of layers: {model.config.n_layer}")

    # Step 4: Extract embeddings
    print(f"\nStep 4: Extracting layer {args.layer} embeddings...")
    embeddings, word_records = extract_embeddings(
        sentences, tokenizer, model, args.layer
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 5: Save
    print("\nStep 5: Saving outputs...")
    np.save(embeddings_path, embeddings)
    pd.DataFrame(word_records).to_csv(word_data_path, index=False)
    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved word data:  {word_data_path}")

    # Step 6: Sanity check
    print("\nSanity check:")
    print(f"  Total words:     {len(word_records)}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Min value:       {embeddings.min():.4f}")
    print(f"  Max value:       {embeddings.max():.4f}")
    print(f"  Mean value:      {embeddings.mean():.4f}")

    print("\nFirst 5 words with embeddings (first 3 dimensions shown):")
    print(f"{'Word':<20} {'Start':>8} {'Emb[0]':>10} {'Emb[1]':>10} {'Emb[2]':>10}")
    print("-" * 62)
    df = pd.DataFrame(word_records)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"{row['word']:<20} {row['start']:>8.3f} "
              f"{embeddings[i,0]:>10.4f} "
              f"{embeddings[i,1]:>10.4f} "
              f"{embeddings[i,2]:>10.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()