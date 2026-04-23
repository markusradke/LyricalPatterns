import pandas as pd

from helpers.STMTopicModeler import STMTopicModeler
from helpers.aggregate_artist_dtm import aggregate_dtm_by_artist
from helpers.load_data import load_stm_data

K_TOPICS = 12
K_SENTIMENT = 8
K_EXPRESSIONS = 15


def fit_and_save_stm_model(
    X_train,
    X_test,
    artists,
    genres,
    vocab,
    name,
    k: int,
) -> None:
    model_dir = f"models/stm_{name}_dc"
    train_out_path = f"data/X_train_{name}_dc.csv"
    test_out_path = f"data/X_test_{name}_dc.csv"

    X_train_artist_agg, genres_agg = aggregate_dtm_by_artist(X_train, artists, genres)
    modeler = STMTopicModeler(
        use_genre_prevalence=True,
        random_state=42,
        model_dir=model_dir,
    )
    modeler.fit(
        k=k,
        X_artist=X_train_artist_agg,
        artist_genres=genres_agg,
        vocab=vocab,
    )

    X_train_transformed = modeler.transform(X_train, vocab)
    X_test_transformed = modeler.transform(X_test, vocab)

    pd.DataFrame(X_train_transformed).to_csv(train_out_path, index=False)
    pd.DataFrame(X_test_transformed).to_csv(test_out_path, index=False)


if __name__ == "__main__":
    (
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
    ) = load_stm_data()

    print("FITTING STM TOPIC MODEL...")
    fit_and_save_stm_model(
        X_train_topics_fighting_full,
        X_test_topics_fighting_full,
        artists,
        genres,
        topics_vocab,
        "topics",
        K_TOPICS,
    )

    print("FITTING STM SENTIMENT MODEL...")
    fit_and_save_stm_model(
        X_train_sentiments_fighting_full,
        X_test_sentiments_fighting_full,
        artists,
        genres,
        sentiments_vocab,
        "sentiments",
        K_SENTIMENT,
    )

    print("FITTING STM EXPRESSIONS MODEL...")
    fit_and_save_stm_model(
        X_train_expressions_fighting_full,
        X_test_expressions_fighting_full,
        artists,
        genres,
        expressions_vocab,
        "expressions",
        K_EXPRESSIONS,
    )
