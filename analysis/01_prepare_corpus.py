import pandas as pd

from helpers.LyricsProcessor import LyricsProcessor
from helpers.language_detection import (
    get_english_confidence,
    get_english_vocab_ratio,
)


def load_raw_corpus(filepath):
    print(f"Loading raw corpus from {filepath}...")
    corpus = pd.read_csv(
        "data-raw/poptrag_lyrics_genres_corpus_20260401.csv",
        delimiter=",",
    )
    cols_to_string = [
        c for c in corpus.columns if not (c.startswith("pmax") or c.startswith("nmax"))
    ]
    corpus[cols_to_string] = corpus[cols_to_string].astype("string")
    return corpus


def filter_tracks(corpus):
    print("Filtering corpus for English tracks with lyrics and genre information...")
    filtered = corpus.copy()
    filtered = filtered[
        (filtered["album.dc.genres"].notna())
        & (filtered["full_lyrics"].notna())
        & (filtered["track.language"].isin(["English"]))
        & (
            filtered["album.s.title"] != "No Grave but the Sea (Deluxe Edition)"
        )  # contains only woof woof
    ]

    filtered["english_conf"] = filtered["full_lyrics"].apply(get_english_confidence)
    filtered["english_vocab_ratio"] = filtered["full_lyrics"].apply(
        get_english_vocab_ratio
    )
    filtered = filtered.query("english_conf > 0.75 and english_vocab_ratio > 0.75")
    return filtered


def assign_and_filter_genres(
    corpus,
    genres_column="album.dc.genres",
    min_track_proportion=0.01,
    genrecol_name="dc_detailed",
):
    print("Assigning detailed genres and filtering by minimum track count...")
    corpus[genres_column] = corpus[genres_column].str.split(";")
    genre_counts = corpus.explode(genres_column)[genres_column].value_counts()
    corpus[genrecol_name] = corpus[genres_column].apply(
        lambda genres: min(genres, key=lambda g: genre_counts[g])
    )
    genre_counts = corpus[genrecol_name].value_counts(normalize=True)
    final_genre_set = genre_counts[genre_counts > min_track_proportion].index
    print(f"Genres retained after filtering: {final_genre_set.tolist()}")
    reduced_corpus = corpus.query(f"{genrecol_name} in @final_genre_set")
    return reduced_corpus


if __name__ == "__main__":
    raw_corpus = load_raw_corpus("data-raw/poptrag_lyrics_genres_corpus_20260401.csv")
    filtered = filter_tracks(raw_corpus)
    reduced_genres = assign_and_filter_genres(filtered)

    print("Processing lyrics for topic, sentiment, and idiomatic analysis...")
    processor = LyricsProcessor(reduced_genres, lyrics_column="full_lyrics")
    final_dataset = processor.process()

    print("Saving consolidated dataset...")
    final_dataset.to_csv("data/poptrag_lyrics_processed.csv", index=True)
    print("Done! Saved to poptrag_lyrics_processed.csv")
    print(f"Columns: {list(final_dataset.columns)}")
