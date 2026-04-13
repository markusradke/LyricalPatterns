import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from helpers.FSExtractor import FSExtractor
from helpers.FightingExtractor import FightingExtractor
from helpers.split_group_stratified_and_join import (
    split_group_stratified_and_join,
    plot_comparison_genre_distributions,
)


RANDOM_STATE = 42
MIN_ARTISTS = 2


def get_and_save_train_test_meta_data():
    print("Loading data and creating train/test split")
    data = pd.read_csv("data/poptrag_lyrics_dc_processed.csv")

    labels_and_groups = data[["dc_detailed", "track.s.firstartist.name"]].rename(
        columns={"dc_detailed": "label", "track.s.firstartist.name": "group"}
    )
    X_train, X_test, y_train, y_test = split_group_stratified_and_join(
        labels_and_groups,
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    fig = plot_comparison_genre_distributions(y_train, y_test)
    fig.savefig(
        "reports/paper_ismir/figures/train_test_genre_distributions.png",
        dpi=1200,
    )

    X_train["expression_lyrics"].to_csv("data/X_train_lyrics_dc.csv", index=False)
    X_test["expression_lyrics"].to_csv("data/X_test_lyrics_dc.csv", index=False)

    X_train_metadata = X_train[
        ["track.s.firstartist.name", "dc_detailed", "track.s.id"]
    ]
    X_train_metadata.to_csv("data/X_train_metadata_dc.csv", index=False)
    X_test_metadata = X_test[["track.s.firstartist.name", "dc_detailed", "track.s.id"]]
    X_test_metadata.to_csv("data/X_test_metadata_dc.csv", index=False)
    return X_train, X_test, y_train, y_test


def extract_and_save_FS_features(X_train, X_test, y_train):
    print("Extracting FS features")
    fs_extractor = FSExtractor(
        min_artists=MIN_ARTISTS,
        use_stopword_filter=False,
        top_vocab_per_genre=100,
        random_state=RANDOM_STATE,
        checkpoint_dir="data/checkpoints/fs_extractor",
    )

    fs_extractor.fit(
        X_train["expression_lyrics"],
        y_train,
        X_train["track.s.firstartist.name"],
    )
    X_train_fs = fs_extractor.transform(X_train["expression_lyrics"])
    X_test_fs = fs_extractor.transform(X_test["expression_lyrics"])
    sparse.save_npz("data/X_train_fs_dc_detailed.npz", X_train_fs)
    sparse.save_npz("data/X_test_fs_dc_detailed.npz", X_test_fs)


def extract_and_save_fighting_vocabularies(X_train, X_test, y_train, lyricscol):
    if lyricscol == "expression_lyrics":
        ngram_types = (1, 2, 3, 4)
        name = "expressions"
        min_char = 0
    else:
        ngram_types = (1,)
        min_char = 3
        if lyricscol == "topic_lyrics":
            name = "topics"
        else:
            name = "sentiments"
    print(f"Extracting Fighting vocabulary for {name} with ngram types {ngram_types}")
    extractor = FightingExtractor(
        min_artists=MIN_ARTISTS,
        p_value=0.001,
        prior_concentration=1.0,
        use_stopword_filter=True,
        ngram_types=ngram_types,
        min_char=min_char,
        random_state=42,
        checkpoint_dir=f"data/checkpoints/fighting_extractor_{name}",
    )
    extractor.fit(X_train[lyricscol], y_train, X_train["track.s.firstartist.name"])

    # Remove empty text placeholder token before saving vocabulary
    vocab = pd.Series([v for v in extractor.vocabulary_ if v != "[empty]"])
    vocab.to_csv(f"data/fighting_{name}_dc_vocabulary.csv", index=False)
    z_scores = extractor.z_scores_df_
    z_scores.to_csv(f"models/fighting_{name}_dc_z_scores.csv", index=False)

    X_train_fighting_full = extractor.transform(X_train[lyricscol])
    X_test_fighting_full = extractor.transform(X_test[lyricscol])

    sparse.save_npz(f"data/X_train_{name}_fighting_dc_full.npz", X_train_fighting_full)
    sparse.save_npz(f"data/X_test_{name}_fighting_dc_full.npz", X_test_fighting_full)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_and_save_train_test_meta_data()
    extract_and_save_FS_features(X_train, X_test, y_train)
    extract_and_save_fighting_vocabularies(X_train, X_test, y_train, "topic_lyrics")
    extract_and_save_fighting_vocabularies(X_train, X_test, y_train, "sentiment_lyrics")
    extract_and_save_fighting_vocabularies(
        X_train, X_test, y_train, "expression_lyrics"
    )
    print("All done!")
