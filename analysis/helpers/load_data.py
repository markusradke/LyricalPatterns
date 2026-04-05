import pandas as pd
import scipy.sparse as sparse


def load_stm_data():
    X_train_metadata = pd.read_csv("data/X_train_metadata_dc.csv")
    genres = X_train_metadata["dc_detailed"]
    artists = X_train_metadata["track.s.firstartist.name"]

    topics_vocab = (
        pd.read_csv("data/fighting_topics_dc_vocabulary.csv").to_numpy().flatten()
    )
    sentiments_vocab = (
        pd.read_csv("data/fighting_sentiments_dc_vocabulary.csv").to_numpy().flatten()
    )
    expressions_vocab = (
        pd.read_csv("data/fighting_expressions_dc_vocabulary.csv").to_numpy().flatten()
    )

    X_train_topics_fighting_full = sparse.load_npz(
        "data/X_train_topics_fighting_dc_full.npz"
    )
    X_test_topics_fighting_full = sparse.load_npz(
        "data/X_test_topics_fighting_dc_full.npz"
    )

    X_train_sentiments_fighting_full = sparse.load_npz(
        "data/X_train_sentiments_fighting_dc_full.npz"
    )
    X_test_sentiments_fighting_full = sparse.load_npz(
        "data/X_test_sentiments_fighting_dc_full.npz"
    )

    X_train_expressions_fighting_full = sparse.load_npz(
        "data/X_train_expressions_fighting_dc_full.npz"
    )
    X_test_expressions_fighting_full = sparse.load_npz(
        "data/X_test_expressions_fighting_dc_full.npz"
    )
    return (
        genres,
        artists,
        topics_vocab,
        sentiments_vocab,
        expressions_vocab,
        X_train_topics_fighting_full,
        X_test_topics_fighting_full,
        X_train_sentiments_fighting_full,
        X_test_sentiments_fighting_full,
        X_train_expressions_fighting_full,
        X_test_expressions_fighting_full,
    )


def load_interpretable_classification_data():
    y_train = pd.read_csv("data/X_train_metadata_dc.csv")["dc_detailed"]
    y_test = pd.read_csv("data/X_test_metadata_dc.csv")["dc_detailed"]

    X_train_fs = sparse.load_npz("data/X_train_fs_dc_detailed.npz")
    X_test_fs = sparse.load_npz("data/X_test_fs_dc_detailed.npz")

    X_train_topics = pd.read_csv("data/X_train_topics_dc.csv").rename(
        columns=lambda x: f"topic_{x}"
    )
    X_test_topics = pd.read_csv("data/X_test_topics_dc.csv").rename(
        columns=lambda x: f"topic_{x}"
    )
    X_train_styles = pd.read_csv("data/X_train_styles_dc.csv").rename(
        columns=lambda x: f"style_{x}"
    )
    X_test_styles = pd.read_csv("data/X_test_styles_dc.csv").rename(
        columns=lambda x: f"style_{x}"
    )
    X_train_combined = pd.concat([X_train_topics, X_train_styles], axis=1)
    X_test_combined = pd.concat([X_test_topics, X_test_styles], axis=1)
    return (
        y_train,
        y_test,
        X_train_fs,
        X_test_fs,
        X_train_topics,
        X_test_topics,
        X_train_styles,
        X_test_styles,
        X_train_combined,
        X_test_combined,
    )
