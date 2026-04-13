library(dplyr)
library(readr)
library(stringr)

topic_vocab <- read_csv("data/fighting_topics_dc_vocabulary.csv") |> pull(`0`)
sentiment_vocab <- read_csv("data/fighting_sentiments_dc_vocabulary.csv") |>
  pull(`0`)
expression_vocab <- read_csv("data/fighting_expressions_dc_vocabulary.csv") |>
  pull(`0`)

length(topic_vocab)
length(sentiment_vocab)
length(expression_vocab)


# get prevalence of ngram types in expressions vocabulary
expression_vocab |> str_count(" ") |> table()
expression_vocab |> str_count(" ") |> table() / length(expression_vocab)
