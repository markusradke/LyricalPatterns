library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(readr)
library(forcats)
library(stm)


# STM INTERPRETATION ----
metadata_train <- read_csv("data/X_train_metadata_dc.csv")
labels_train <- metadata_train$dc_detailed
metadata_test <- read_csv("data/X_test_metadata_dc.csv")
labels_test <- metadata_test$dc_detailed


save_top_frex_examples <- function(model, labels, dimension) {
  top_frex <- labelTopics(model, n = 200)$frex
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

prep_mean_props_by_genre <- function(thetas, labels, genres) {
  thetas$dc_detailed <- genres
  pivoted_thetas <- thetas |>
    pivot_longer(
      cols = all_of(labels),
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
      axis.text.x = element_text(angle = 80, hjust = 0, vjust = 0, size = 16),
      axis.text.y = element_text(size = 16),
      legend.position = "none",
      plot.margin = margin(0, 1, 0, 0, unit = "cm")
    )
}

# inerpetation of topic model ----
topic_model <- readRDS("models/stm_topics_dc/stm_model.rds")


labelTopics(topic_model, n = 50)$frex[7, ]

topic_labels <- c(
  "Americana",
  "Apocalypse",
  "Existentialism",
  "Introspection",
  "Working Life",
  "Street Life",
  "Urban Narrative",
  "Romance",
  "Nature & Holidays",
  "Struggle",
  "Sex & Nightlife",
  "Heroic Tales"
)
topic_ref_type <- which.max(topic_model$theta |> colSums())
topic_labels[topic_ref_type]

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
labelTopics(sentiments_model, n = 50)$frex[8, ]

sentiment_labels <- c(
  "delightful",
  "apocalyptic",
  "descriptive", # ambivalent?
  "otherwordly", # etherial-dark
  "nostalgic",
  "melancholic", # ambivalent?
  "vulnerable",
  "irreverent" # rude, brash, crude
)
sentiment_ref_type <- which.max(sentiments_model$theta |> colSums())
sentiment_labels[sentiment_ref_type]

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
labelTopics(expressions_model, n = 200)$frex[13, ]

expressions_labels <- c(
  "Love Affirmations",
  "Depressing Expressions",
  "Dark Fantasy Imagery",
  "Jamaican Patois & Gospel",
  "Old School Street Slang",
  "Sexually-charged Partying",
  "Denying & Negotiating",
  "Violent Threats & Insults",
  "Street Boasting",
  "Storytelling",
  "Vintage Nostalgia",
  "Dance Floor Shoutings",
  "Longing & Nature Imagery",
  "Christmas & Traditionals",
  "Struggle Expressions"
)
expresssions_ref_type <- which.max(expressions_model$theta |> colSums())
expressions_labels[expresssions_ref_type]
ref_types <- data.frame(
  dimension = c("topics", "sentiments", "expressions"),
  base = c(topic_ref_type, sentiment_ref_type, expresssions_ref_type) - 1
)
write_csv(ref_types, "models/ref_types_zero_indexed.csv")

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
# thetas_X_train_expressions <- read_csv("data/X_train_expressions_dc_quickfix.csv")
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

# ngram type distribution ----
vocab_stats <- read_csv(
  "data/checkpoints/fighting_extractor_expressions/vocab_stats_f1ee054c34695f6f30e64d80670afe91.csv"
) |>
  filter(str_detect(type, "Types")) |>
  mutate(
    type = str_extract(type, ".*(?= Types)"),
    relfreq = ifelse(
      round(relfreq, 2) >= 0.01,
      sprintf("%d%%", round(relfreq * 100)),
      sprintf("%d%% (%d n-grams)", round(relfreq * 100), freq)
    )
  ) |>
  glimpse()

total_types <- vocab_stats |> filter(type == "Total") |> pull(freq)
vocab_stats <- vocab_stats |> filter(type != "Total")
ggplot(vocab_stats, aes(y = fct_inorder(type) |> fct_rev(), x = freq)) +
  geom_col(fill = "grey65") +
  theme_minimal() +
  geom_text(aes(label = relfreq), hjust = -0.1) +
  scale_x_continuous(
    limits = c(0, total_types),
    breaks = c(seq(0, 10000, 2000), seq(15000, 25000, 5000)),
    labels = c("0", sprintf("%dk", c(seq(2, 10, 2), 15, 20, 25))),
    expand = expansion(mult = c(0, 0.05))
  ) +
  geom_vline(xintercept = total_types, linetype = "dashed", color = "#c40d20") +
  annotate(
    "text",
    label = "total unique n-grams",
    x = total_types,
    y = 1,
    hjust = 1.1,
    vjust = 0.5,
    size = 3,
    color = "#c40d20"
  ) +
  ylab(label = "") +
  xlab(label = "Absolute Frequency") +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14, )
  )
ggsave(
  "reports/paper_ismir/figures/ngram_type_distribution.png",
  width = 6,
  height = 3
)
