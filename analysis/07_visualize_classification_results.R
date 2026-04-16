library(ggplot2)
library(dplyr)
library(readr)
library(forcats)


# Confusion Matrix and variable importance
y_pred <- read_csv(
  "models/lgbm_combined/y_pred.csv"
)$pred |>
  as.factor()
y_test <- read_csv("data/X_test_metadata_dc.csv")$dc_detailed |>
  as.factor()
conf_matrix <- table(y_test, y_pred)
conf_matrix_long <- as.data.frame(as.table(conf_matrix))

# caluclate genre F1-scores
f1_scores <- sapply(levels(y_test), function(genre) {
  tp <- conf_matrix[genre, genre]
  fp <- sum(conf_matrix[, genre]) - tp
  fn <- sum(conf_matrix[genre, ]) - tp
  if (tp == 0 || tp + fp == 0 || tp + fn == 0) {
    return(0) # Avoid division by zero and set to 0 if there are no positive predictions
  }
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(f1)
})

# correlation between number of training samples and F1-scores
X_train_metadata <- read_csv("data/X_train_metadata_dc.csv")
train_genre_counts <- X_train_metadata |>
  count(dc_detailed) |>
  rename(genre = dc_detailed, count = n)
f1_scores_df <- data.frame(
  genre = names(f1_scores),
  f1_score = f1_scores
) |>
  left_join(train_genre_counts, by = "genre")
f1_samples_cor <- cor(
  f1_scores_df$count,
  f1_scores_df$f1_score,
  method = "kendall"
)

metrics_label <- sprintf(
  "F1<sub>macro</sub>: %.3f<br>
  Cor<sub>Kendall</sub>(F1, # samples): %.3f",
  0.300,
  f1_samples_cor
)

# transform to relative frequencies
conf_matrix_labelled <- conf_matrix_long |>
  group_by(y_test) |>
  mutate(
    Freq = Freq / sum(Freq),
    f1_score = f1_scores[match(y_test, levels(y_test))],
    y_test = y_test |>
      as.character() |>
      paste0(" (F1: ", sprintf("%.3f", f1_score), ")") |>
      as.factor()
  )

ggplot(
  conf_matrix_labelled,
  aes(factor(y_pred), fct_rev(y_test), fill = Freq)
) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "#c40d20") +
  scale_x_discrete(position = "top") +
  geom_text(
    aes(label = sprintf("%.0f", Freq * 100)),
    color = ifelse(conf_matrix_labelled$Freq > 0.5, "white", "black"),
    size = 4
  ) +
  labs(x = "Predicted", y = "Target", title = metrics_label) +
  theme_minimal() +
  theme(
    plot.title = ggtext::element_markdown(
      size = 16,
      face = "bold",
      color = "black",
      hjust = 0,
      lineheight = 1.5,
      margin = ggplot2::margin(b = -41.5, l = -200)
    ),
    axis.text.x = element_text(angle = 45, hjust = 0, vjust = 0, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.position = "none"
  )
ggsave(
  "reports/paper_ismir/figures/confusion_matrix.png",
  width = 6,
  height = 5
)

# make a plot for the permutation importance
prepare_importance <- function(filepath) {
  pfi <- read_csv(filepath) |>
    arrange(-mean_importance) |>
    mutate(
      feature_group = case_when(
        grepl("topic", feature) ~ "topics",
        grepl("sentiment", feature) ~ "sentiments",
        grepl("expression", feature) ~ "expressions"
      ),
      feature_group = factor(
        feature_group,
        levels = c("topics", "sentiments", "expressions")
      )
    )
  pfi$feature <- feature_name_lookup[pfi$feature]
  pfi$feature = fct_inorder(pfi$feature) |> fct_rev()
  return(pfi)
}

plot_pfi <- function(pfi, name = "validation") {
  ggplot(pfi, aes(y = feature, x = mean_importance, fill = feature_group)) +
    geom_col() +
    geom_errorbar(
      aes(
        xmin = mean_importance - std_importance,
        xmax = mean_importance + std_importance
      ),
      width = 0.2
    ) +
    labs(x = sprintf("Mean Impact on %s F1 macro", name), y = "Feature") +
    scale_x_continuous(expand = expansion(mult = (c(0, 0.55)))) +
    scale_fill_manual(
      breaks = c("topics", "sentiments", "expressions"),
      values = c(
        "topics" = "#c40d20",
        "sentiments" = "#2b8dba",
        "expressions" = "grey65"
      )
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.8, 0.2),
      legend.title = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 12),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14)
    )
}


topic_labels <- read_csv("data/topic_labels.csv") |>
  mutate(feature_label = paste0("topic_", topic_num))
topics_lookup <- topic_labels$topic_labels
names(topics_lookup) <- topic_labels$feature_label
sentiment_labels <- read_csv("data/sentiment_labels.csv") |>
  mutate(feature_label = paste0("sentiment_", topic_num))
sentiments_lookup <- sentiment_labels$sentiment_labels
names(sentiments_lookup) <- sentiment_labels$feature_label
expressions_labels <- read_csv("data/expressions_labels.csv") |>
  mutate(feature_label = paste0("expressions_", topic_num))
expressions_lookup <- expressions_labels$expressions_labels
names(expressions_lookup) <- expressions_labels$feature_label
feature_name_lookup <- c(topics_lookup, sentiments_lookup, expressions_lookup)


pfi_val <- prepare_importance("models/lgbm_combined/permutation_importance.csv")
pfi_holdout <- prepare_importance(
  "models/lgbm_combined/holdout_permutation_importance.csv"
)

plot_pfi(pfi_val, "validation")
ggsave(
  "reports/paper_ismir/figures/permutation_importance.png",
  width = 6,
  height = 5
)

plot_pfi(pfi_holdout, "holdout")
ggsave(
  "reports/paper_ismir/figures/holdout_permutation_importance.png",
  width = 6,
  height = 5
)
