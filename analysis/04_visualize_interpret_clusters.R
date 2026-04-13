library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stm)


# STM INTERPRETATION ----
metadata <- read_csv("data/X_train_metadata_dc.csv")
labels <- metadata$dc_detailed
# aggregate lyrics by artist (sort alphabetically) for inspection
dc_data <- read_csv(
  "data/poptrag_lyrics_dc_processed.csv"
)
agg_lyrics <- dc_data |>
  filter(track.s.id %in% metadata$track.s.id) |>
  group_by(track.s.firstartist.name) |>
  summarize(
    lyrics = paste(lyrics_lemmatized, collapse = " "),
    .groups = "drop"
  ) |>
  arrange(track.s.firstartist.name) |>
  pull(lyrics)

print_thoughts <- function(model, topic_num, n_docs = 5, max_chars = 500) {
  toughts <- stm::findThoughts(model, texts = agg_lyrics, n = n_docs)
  cat(paste0("TOPIC ", topic_num, ":\n"))
  for (j in 1:n_docs) {
    cat(paste0("Artist ", j, ":\n"))
    cat(
      substr(toughts$docs[[topic_num]][[j]], 1, max_chars),
      "\n"
    )
    cat("\n------------------------------\n")
  }
  cat("\n==============================\n")
  cat("\n\n")
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
print_thoughts(topic_model, n_docs = 5, topic_num = 11)

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

stm::plot.STM(
  topic_model,
  type = "hist",
  labeltype = "frex",
  topic.names = topic_labels
)
topic_corr <- stm::topicCorr(topic_model, method = "simple", cutoff = 0.01)
stm::plot.topicCorr(topic_corr, vlabels = topic_labels)

# heatmap for paper
thetas_X_train_topics <- read_csv("data/X_train_topics_dc.csv")
colnames(thetas_X_train_topics) <- topic_labels
pivoted_thetas_topics <- prep_mean_props_by_genre(
  thetas_X_train_topics,
  topic_labels,
  labels
)
plot_heatmap(pivoted_thetas_topics)
ggsave(
  "reports/paper_ismir/figures/topic_genre_heatmap.png",
  width = 6,
  height = 5.5
)

# interpretation of sentiment model ----
sentiments_model <- readRDS("models/stm_sentiments_dc/stm_model.rds")
# interpret only LIFT scores (betonung auf uniqueness)
labelTopics(sentiments_model, n = 50)$frex[3, ]
print_thoughts(sentiments_model, n_docs = 5, topic_num = 13)

sentiment_labels <- c(
  "playful", # irreverent?
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
  labels
)
plot_heatmap(pivoted_thetas_sentiment)
ggsave(
  "reports/paper_ismir/figures/sentiment_genre_heatmap.png",
  width = 6,
  height = 5.5
)


# interpretation of expressions model ----
# USE FREX FOR INTERPRETATION
retain_maximal_phrases <- function(phrases) {
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

expressions_model <- readRDS("models/stm_expressions_dc/stm_model.rds")
(labelTopics(expressions_model, n = 100)$frex[8, ] |>
  retain_maximal_phrases())[
  1:50
]


expressions_labels <- c(
  "Denials & Negotiations",
  "Traditional Street Slang",
  "Nostalgic Formulas",
  "Trap Slang",
  "Christmas Phrases",
  "Doom Imagery",
  "Love Affirmations",
  "Dance Floor Commands",
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
  labels
)
plot_heatmap(pivoted_thetas_expressions)
ggsave(
  "reports/paper_ismir/figures/expressions_genre_heatmap.png",
  width = 6,
  height = 5.5
)
