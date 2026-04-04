library(ggplot2)
library(ggrepel) # TODO: use repel labels for genres in the area plot
library(dplyr)
library(readr)
library(forcats)
library(Polychrome)


data("Dark24", package = "Polychrome")
GENRE_COLORS <- Dark24

# CORPUS DESCRIPTIVES ----
dc_data <- read_csv(
  "data/poptrag_lyrics_dc_processed.csv"
)
dplyr::glimpse(dc_data)
count_data <- count(dc_data, dc_detailed) |>
  mutate(relfreq = n / sum(n) * 100)
dc_data_labelled <- dc_data %>%
  left_join(count_data, by = "dc_detailed") %>%
  mutate(dc_detailed = paste0(dc_detailed, " (", round(relfreq, 1), "%)"))

# make labels a factor according to the number of tracks
dc_data_labelled$dc_detailed <- factor(
  dc_data_labelled$dc_detailed,
  levels = count_data$dc_detailed[order(count_data$n, decreasing = TRUE)] %>%
    paste0(
      " (",
      round(count_data$relfreq[order(count_data$n, decreasing = TRUE)], 1),
      "%)"
    )
)
dc_data_labelled$dc_detailed <- fct_rev(dc_data_labelled$dc_detailed)

# plot area by genre over years
ggplot(dc_data_labelled, aes(x = album.s.releaseyear, fill = dc_detailed)) +
  geom_area(stat = "count") +
  labs(
    x = "Album Release Year",
    y = "Number of Tracks"
  ) +
  scale_fill_discrete(palette = GENRE_COLORS, name = "Discogs") +
  scale_y_continuous(
    breaks = seq(0, 5000, by = 1000),
    labels = c(0, sprintf("%dk", seq(5))),
  ) +
  guides(fill = guide_legend(nrow = 5)) +
  theme_minimal() +
  xlim(1960, 2024) +
  theme(
    legend.position = "top",
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )
ggsave(
  "reports/paper_ismir/figures/corpus_descriptive.png",
  width = 6,
  height = 5,
)
