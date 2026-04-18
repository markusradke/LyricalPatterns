library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stm)


# STM INTERPRETATION ----
metadata_train <- read_csv("data/X_train_metadata_dc.csv")
labels_train <- metadata_train$dc_detailed
metadata_test <- read_csv("data/X_test_metadata_dc.csv")
labels_test <- metadata_test$dc_detailed


retain_maximal_phrases <- function(phrases) {
  # Returns a subset of phrases where no phrase is a contiguous substring of another.
  # For interpretation of expressions model
  is_substring_of <- function(a, b) {
    words_a <- strsplit(a, " ")[[1]]
    words_b <- strsplit(b, " ")[[1]]

    if (length(words_a) >= length(words_b)) {
      return(FALSE)
    }

    len_a <- length(words_a)
    for (i in 1:(length(words_b) - len_a + 1)) {
      if (all(words_b[i:(i + len_a - 1)] == words_a)) {
        return(TRUE)
      }
    }
    return(FALSE)
  }

  indices <- seq_along(phrases)
  keep <- rep(TRUE, length(phrases))

  for (i in indices) {
    for (j in indices) {
      if (i != j && keep[i] && keep[j]) {
        if (is_substring_of(phrases[i], phrases[j])) {
          keep[i] <- FALSE
        }
      }
    }
  }

  phrases[keep]
}


save_top_frex_examples <- function(model, labels, dimension) {
  if (dimension == "expressions") {
    top_frex <- matrix(
      NA,
      nrow = ncol(model$theta),
      ncol = 50
    )
    for (i in 1:nrow(top_frex)) {
      top_frex[i, ] <- retain_maximal_phrases(
        labelTopics(model, n = 200)$frex[i, ]
      )[1:ncol(top_frex)]
    }
  } else {
    top_frex <- labelTopics(model, n = 50)$frex
  }
  top_frex_df <- data.frame(
    type_num = rep(0:(nrow(top_frex) - 1), each = ncol(top_frex)),
    type_label = rep(labels, each = ncol(top_frex)),
    ngram_rank = rep(1:ncol(top_frex), times = nrow(top_frex)),
    ngram = as.vector(t(top_frex))
  )
  filepath <- sprintf(
    "models/stm_%s_dc/%s_top_frex_words.csv",
    dimension,
    dimension
  )
  readr::write_csv(top_frex_df, filepath)
}

prep_mean_props_by_genre <- function(thetas, topic_labels, genres) {
  thetas$dc_detailed <- genres
  pivoted_thetas <- thetas |>
    pivot_longer(
      cols = all_of(topic_labels),
      names_to = "variable",
      values_to = "value"
    ) |>
    group_by(dc_detailed, variable) |>
    summarize(value = mean(value), .groups = "drop")
}

plot_heatmap <- function(pivoted_thetas) {
  ggplot(pivoted_thetas, aes(x = variable, y = dc_detailed, fill = value)) +
    geom_tile() +
    scale_fill_gradient(limits = c(0, 0.5), low = "white", high = "#c40d20") +
    geom_text(
      aes(label = sprintf("%.0f", value * 100)),
      color = ifelse(pivoted_thetas$value > 0.2, "white", "black"),
      size = 4.5
    ) +
    scale_x_discrete(position = "top") +
    labs(x = "", y = "") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 70, hjust = 0, vjust = 0, size = 16),
      axis.text.y = element_text(size = 16),
      legend.position = "none",
      plot.margin = margin(0, 1, 0, 0, unit = "cm")
    )
}

# inerpetation of topic model ----
topic_model <- readRDS("models/stm_topics_dc/stm_model.rds")

labelTopics(topic_model, n = 50)$frex[2, ]

topic_labels <- c(
  "Heroic Saga",
  "Darkness", # Apocalyptic?
  "Embodiment",
  "Americana",
  "Holidays",
  "Street Life",
  "Romance",
  "Nature",
  "Narrative",
  "Existentialism"
)
readr::write_csv(
  data.frame(topic_labels, topic_num = 0:(length(topic_labels) - 1)),
  "data/topic_labels.csv"
)

save_top_frex_examples(topic_model, topic_labels, "topics")


stm::plot.STM(
  topic_model,
  type = "hist",
  labeltype = "frex",
  topic.names = topic_labels
)
topic_corr <- stm::topicCorr(topic_model, method = "simple", cutoff = 0.01)
stm::plot.topicCorr(topic_corr, vlabels = topic_labels)

# heatmaps
thetas_X_train_topics <- read_csv("data/X_train_topics_dc.csv")
colnames(thetas_X_train_topics) <- topic_labels
pivoted_thetas_train_topics <- prep_mean_props_by_genre(
  thetas_X_train_topics,
  topic_labels,
  labels_train
)
plot_heatmap(pivoted_thetas_train_topics)
ggsave(
  "reports/paper_ismir/figures/topic_genre_heatmap_train.png",
  width = 6,
  height = 5.5
)

thetas_X_test_topics <- read_csv("data/X_test_topics_dc.csv")
colnames(thetas_X_test_topics) <- topic_labels
pivoted_thetas_test_topics <- prep_mean_props_by_genre(
  thetas_X_test_topics,
  topic_labels,
  labels_test
)
plot_heatmap(pivoted_thetas_test_topics)
ggsave(
  "reports/paper_ismir/figures/topic_genre_heatmap_holdout.png",
  width = 6,
  height = 5.5
)


# interpretation of sentiment model ----
sentiments_model <- readRDS("models/stm_sentiments_dc/stm_model.rds")
# interpret only LIFT scores (betonung auf uniqueness)
labelTopics(sentiments_model, n = 50)$frex[1, ]

sentiment_labels <- c(
  "affirmative", # irreverent?
  "nostalgic", # melancholic?
  "apocalyptic", # dark
  "romantic", # affectionate
  "brash", # explicit, irreverent, bragging
  "ambivalent" #?
)
readr::write_csv(
  data.frame(sentiment_labels, topic_num = 0:(length(sentiment_labels) - 1)),
  "data/sentiment_labels.csv"
)
save_top_frex_examples(sentiments_model, sentiment_labels, "sentiments")

stm::plot.STM(
  sentiments_model,
  type = "hist",
  labeltype = "frex",
  topic.names = sentiment_labels
)
sentiment_corr <- stm::topicCorr(
  sentiments_model,
  method = "simple",
  cutoff = 0.01
)
stm::plot.topicCorr(sentiment_corr, vlabels = sentiment_labels)

thetas_X_train_sentiment <- read_csv("data/X_train_sentiments_dc.csv")
colnames(thetas_X_train_sentiment) <- sentiment_labels
pivoted_thetas_sentiment <- prep_mean_props_by_genre(
  thetas_X_train_sentiment,
  sentiment_labels,
  labels_train
)
plot_heatmap(pivoted_thetas_sentiment)
ggsave(
  "reports/paper_ismir/figures/sentiment_genre_heatmap_train.png",
  width = 6,
  height = 5.5
)

thetas_X_test_sentiment <- read_csv("data/X_test_sentiments_dc.csv")
colnames(thetas_X_test_sentiment) <- sentiment_labels
pivoted_thetas_test_sentiment <- prep_mean_props_by_genre(
  thetas_X_test_sentiment,
  sentiment_labels,
  labels_test
)
plot_heatmap(pivoted_thetas_test_sentiment)
ggsave(
  "reports/paper_ismir/figures/sentiment_genre_heatmap_holdout.png",
  width = 6,
  height = 5.5
)


# interpretation of expressions model ----
expressions_model <- readRDS("models/stm_expressions_dc/stm_model.rds")
(labelTopics(expressions_model, n = 200)$frex[8, ] |>
  retain_maximal_phrases())[
  1:50
]


expressions_labels <- c(
  "Denials & Doubts", # ist nicht so recht ein Typ expression, oder?
  "Hip Hop & Street Slang", # Hip Hop and Street Slang?
  "Blues & Rock Idioms", # hmm...passt? Ist doch auch anderes
  "Trap Slang",
  "Christmas Phrases",
  "Doom Imagery",
  "Love Affirmations",
  "Dance Floor Shoutings",
  "Landscape Imaginary",
  "Vocalizations",
  "Jamaican Patois"
)
readr::write_csv(
  data.frame(
    expressions_labels,
    topic_num = 0:(length(expressions_labels) - 1)
  ),
  "data/expressions_labels.csv"
)
save_top_frex_examples(
  expressions_model,
  expressions_labels,
  "expressions"
)

stm::plot.STM(
  expressions_model,
  type = "hist",
  labeltype = "frex",
  topic.names = expressions_labels
)
expressions_corr <- stm::topicCorr(
  expressions_model,
  method = "simple",
  cutoff = 0.01
)
stm::plot.topicCorr(expressions_corr, vlabels = expressions_labels)

thetas_X_train_expressions <- read_csv("data/X_train_expressions_dc.csv")
colnames(thetas_X_train_expressions) <- expressions_labels
pivoted_thetas_expressions <- prep_mean_props_by_genre(
  thetas_X_train_expressions,
  expressions_labels,
  labels_train
)
plot_heatmap(pivoted_thetas_expressions)
ggsave(
  "reports/paper_ismir/figures/expressions_genre_heatmap_train.png",
  width = 6,
  height = 5.5
)

thetas_X_test_expressions <- read_csv("data/X_test_expressions_dc.csv")
colnames(thetas_X_test_expressions) <- expressions_labels
pivoted_thetas_test_expressions <- prep_mean_props_by_genre(
  thetas_X_test_expressions,
  expressions_labels,
  labels_test
)
plot_heatmap(pivoted_thetas_test_expressions)
ggsave(
  "reports/paper_ismir/figures/expressions_genre_heatmap_holdout.png",
  width = 6,
  height = 5.5
)
