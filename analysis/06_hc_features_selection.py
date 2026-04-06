from networkx.algorithms.bipartite import color
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif

from helpers.load_data import load_interpretable_classification_data

# necessary because feature importance from tree-based models can be misleading when features are correlated; we want to pick one representative per cluster of highly correlated features, and we want to pick the one with the highest MI with the target
OPTIMAL_N_CLUSTERS = 35


def get_mututal_information_scores(X_train, y_train):
    print("Computing mutual information scores for all features...")
    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features="auto",  # X_train_combiend fully numeric
        n_neighbors=3,
        random_state=42,
    )
    mi_series = pd.Series(mi_scores, index=X_train.columns, name="mi")
    return mi_series


def perform_hc_and_tune_K(X_train_combined):
    print("Performing hierarchical clustering based on Spearman Correlation...")
    corr = spearmanr(X_train_combined).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    silhouette_scores = []
    for n_clusters in range(2, 100):
        cluster_labels = hierarchy.fcluster(
            dist_linkage, n_clusters, criterion="maxclust"
        )
        score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        silhouette_scores.append(score)
    return dist_linkage, corr, silhouette_scores


def get_features_for_chosen_K(distlinkage, k):
    print(f"Extracting representative features for K={k} clusters...")
    cluster_labels = hierarchy.fcluster(distlinkage, k, criterion="maxclust")
    representatives = []
    for cluster in np.unique(cluster_labels):
        cluster_features = X_train_combined.columns[cluster_labels == cluster]
        best_feature = mi_series[cluster_features].idxmax()
        representatives.append(best_feature)
    threshold = distlinkage[len(distlinkage) - k, 2]
    return cluster_labels, representatives, threshold


def save_silhouette_scores_plot(silhouette_scores):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(2, 100), silhouette_scores, marker="o")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Scores for Different Cluster Counts")
    ax.axvline(
        x=OPTIMAL_N_CLUSTERS,
        color="red",
        linestyle="--",
        label="Chosen Number of Clusters",
    )
    ax.grid()
    fig.tight_layout()
    fig.savefig("models/hc_features_selection/silhouette_scores.png", dpi=1200)


def save_dendogram_plot(dist_linkage, threshold, X_train):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    dendro = hierarchy.dendrogram(
        dist_linkage,
        labels=X_train.columns.to_list(),
        ax=ax1,
        leaf_rotation=90,
        color_threshold=threshold,
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    ax1.axhline(y=threshold, color="red", linestyle="--")

    ax2.imshow(
        corr[dendro["leaves"], :][:, dendro["leaves"]], cmap="coolwarm", vmin=-1, vmax=1
    )
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    fig.savefig("models/hc_features_selection/dendrogram.png", dpi=1200)


def save_selected_features_plots(X_train, representatives):
    selected_corr = spearmanr(X_train[representatives]).correlation
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(selected_corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(representatives)))
    ax.set_yticks(np.arange(len(representatives)))
    ax.set_xticklabels(representatives, rotation=90)
    ax.set_yticklabels(representatives)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(
        "models/hc_features_selection/selected_features_correlation.png", dpi=1200
    )

    # get histogram of correlation values to check overall correlation structure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(corr[np.triu_indices_from(corr, k=1)], bins=20)
    ax.set_xlabel("Spearman Correlation")
    ax.set_title("Correlation Distribution of Selected Features")
    fig.tight_layout()
    fig.savefig(
        "models/hc_features_selection/selected_features_correlation_histogram.png",
        dpi=1200,
    )


if __name__ == "__main__":
    print("Loading data...")
    (
        y_train,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        X_train_combined,
        _,
    ) = load_interpretable_classification_data()
    mi_series = get_mututal_information_scores(X_train_combined, y_train)
    dist_linkage, corr, silhouette_scores = perform_hc_and_tune_K(X_train_combined)
    cluster_lables, representatives, threshold = get_features_for_chosen_K(
        dist_linkage, OPTIMAL_N_CLUSTERS
    )
    if not os.path.exists("models/hc_features_selection"):
        os.makedirs("models/hc_features_selection")
    pd.Series(representatives, name="features").to_csv(
        "models/hc_features_selection/representative_features.csv", index=False
    )
    print("Saving plots...")
    save_dendogram_plot(dist_linkage, threshold, X_train_combined)
    save_silhouette_scores_plot(silhouette_scores)
    save_selected_features_plots(X_train_combined, representatives)
    print("All done!")
