import numpy as np

from helpers.STMTopicModeler import STMTopicModeler
from helpers.aggregate_artist_dtm import aggregate_dtm_by_artist
from helpers.load_data import load_stm_data

# K_GRID =  [x for x in range(2,21)]

# K_GRID = [2] + [x for x in range(5, 100) if x % 5 == 0]
K_GRID = [3]


def tune_stm_model(
    X_train,
    artists,
    genres,
    vocab,
    name: str,
) -> None:
    model_dir = f"models/stm_{name}_dc"
    X_train_artist_agg, genres_agg = aggregate_dtm_by_artist(X_train, artists, genres)
    modeler = STMTopicModeler(
        use_genre_prevalence=True,
        random_state=42,
        model_dir=model_dir,
    )
    modeler.search_k_with_heldout(
        X_train_artist_agg,
        genres_agg,
        vocab,
        K_GRID,
    )


if __name__ == "__main__":

    (
        genres,
        artists,
        topics_vocab,
        sentiments_vocab,
        expressions_vocab,
        X_train_topics_fighting_full,
        _,
        X_train_sentiments_fighting_full,
        _,
        X_train_expressions_fighting_full,
        _,
    ) = load_stm_data()

    print("TUNING STM TOPIC MODEL...")
    tune_stm_model(
        X_train_topics_fighting_full, artists, genres, topics_vocab, "topics"
    )

    print("TUNING STM SENTIMENTS MODEL...")
    tune_stm_model(
        X_train_sentiments_fighting_full,
        artists,
        genres,
        sentiments_vocab,
        "sentiments",
    )

    print("TUNING STM EXPRESSIONS MODEL...")
    tune_stm_model(
        X_train_expressions_fighting_full,
        artists,
        genres,
        expressions_vocab,
        "expressions",
    )
