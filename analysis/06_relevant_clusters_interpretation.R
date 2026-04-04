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
  "data/poptrag_lyrics_genres_corpus_filtered_english_dc_lemmatized.csv"
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
  ggplot(pivoted_thetas, aes(x = dc_detailed, y = variable, fill = value)) +
    geom_tile() +
    scale_fill_gradient(limits = c(0, 0.4), low = "white", high = "#c40d20") +
    geom_text(
      aes(label = sprintf("%.0f", value * 100)),
      color = ifelse(pivoted_thetas$value > 0.2, "white", "black"),
      size = 4
    ) +
    scale_x_discrete(position = "top") +
    labs(x = "", y = "") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 0, vjust = 0, size = 12),
      axis.text.y = element_text(size = 12),
      legend.position = "none"
    )
}

# inerpetation of topic model ----
topic_model <- readRDS("models/stm_topics_dc/stm_model.rds")

labelTopics(topic_model, n = 50)
print_thoughts(topic_model, n_docs = 5, topic_num = 11)
stm::plot.STM(topic_model, type = "summary", labeltype = "frex")
stm::plot.STM(topic_model, type = "summary", labeltype = "lift")
stm::plot.STM(topic_model, type = "summary", labeltype = "score")

topic_labels <- c(
  "Gore", #0 Carnage, Graphic Violence, Death
  "Everyday Narrative", #1 Locality, everyday narrative and mundane social observation
  "Nature Reverie", #2 ethereal, poetic, and introspective content centered on natural imagery and luminiscence
  "Black Culture", #3 performance, rhythm, and participatory dance culture
  "Melodics", #4 non-lexical vocalization and melodic repetition
  "Rastafarian identity", # 5 Jamaican dialect with cultural and spiritual content
  "Nightlife", # 6 Euphoria Dance, Raving, ecstatic, and euphoric dance culture; Nightlife
  "Scatting", # 7 sonics, non-lexical vocalizations and vocal sounds, more playful
  "Mythology / Fantasy", # 8 epic warfare, mythological/fantastical conflict, and apocalyptic spirituality, Fantasy, Valor
  "Romantic Love", # 9 romantic love, emotional vulnerability, and longing
  "Internal Conflict", # 10  uncertainty, ambivalence, and emotional ambivalence
  "Sex", # 11 Women, Sexual Attraction
  "Braggadocio", # 12 Street, dominance, aggressive street discourse and confrontational posturing
  "Resignation", # 13 loss, no return, reflection
  "Christmas", # 14
  "Nostalgia" # 15
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

# interpretation of style model ----
style_model <- readRDS("models/stm_styles_dc/stm_model.rds")
# interpret only LIFT scores (betonung auf uniqueness)
labelTopics(style_model, n = 10)
print_thoughts(style_model, n_docs = 5, topic_num = 13)
stm::plot.STM(style_model, type = "summary", labeltype = "frex")
stm::plot.STM(style_model, type = "summary", labeltype = "lift")
stm::plot.STM(style_model, type = "summary", labeltype = "score")

style_labels <- c(
  "Festive / Yuletide", #1
  "Epic / Mythic", #2
  "Urban / Street", #3
  "Na na na", #4 Playful Vocalizations?
  "Nostalgic", #5
  "Romantic / Tender", # 6
  "Weathered / Spritual", # 7
  "Rastafarian", # 8
  "Oh / ooh / la", # 9 Expressive Vocalizations?
  "Etheral / Dreamy", # 10
  "Somber / Introspective", # 11
  "Urban / Aggressive", # 12
  "Conversational", # 13
  "Uncertain / Restless", # 14
  "Extremely Violent" # 15
)
readr::write_csv(
  data.frame(style_labels, topic_num = 0:(length(style_labels) - 1)),
  "data/style_labels.csv"
)

stm::plot.STM(
  style_model,
  type = "hist",
  labeltype = "frex",
  topic.names = style_labels
)
style_corr <- stm::topicCorr(style_model, method = "simple", cutoff = 0.01)
stm::plot.topicCorr(style_corr, vlabels = style_labels)

thetas_X_train_style <- read_csv("data/X_train_style_dc.csv")
colnames(thetas_X_train_style) <- style_labels
pivoted_thetas_style <- prep_mean_props_by_genre(
  thetas_X_train_style,
  style_labels,
  labels
)
plot_heatmap(pivoted_thetas_style)
ggsave(
  "reports/paper_ismir/figures/style_genre_heatmap.png",
  width = 6,
  height = 5.5
)

# Correlation Matrix between topics and styles, each topic with each style
topics <- thetas_X_train_topics
styles <- thetas_X_train_style
cor_matrix <- cor(topics, styles)
cor_matrix_long <- as.data.frame(as.table(cor_matrix))

ggplot(cor_matrix_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "#c40d20") +
  scale_x_discrete(position = "top") +
  geom_text(
    aes(label = sprintf("%.2f", Freq)),
    color = ifelse(cor_matrix_long$Freq > 0.7, "white", "black"),
    size = 3
  ) +
  labs(x = "Topics", y = "Styles") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 55, hjust = 0, vjust = 0, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.position = "none",
    plot.margin = margin(0, 1.5, 0, 0, "cm")
  )
ggsave(
  "reports/paper_ismir/figures/topic_style_correlation.png",
  width = 7,
  height = 6
)
