library(dplyr)
library(readr)
library(parsnip)
library(rules)
library(yardstick)

simplifynames <- function(x) {
  tolower(gsub("[^a-zA-Z0-9]", "_", x))
}

topic_labels <- c(
  "Existential Reflection", #1
  "Vocalizations", #2
  "Old School Gangsta", #3
  "Americana", #4
  "Christmas / Reggae", #5
  "Dark Despair", # 6
  "Desire / Heartbreak", # 7
  "Romantic Love", # 8
  "Nature Imaginary", # 9
  "Hip Hop Slang", # 10
  "Women and Body", # 11
  "Mythic Doom", # 12
  "Wild Fast Living" # 13
) |>
  simplifynames()

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
) |>
  simplifynames()

train_metadata <- read_csv("data/X_train_metadata_dc.csv")
test_metadata <- read_csv("data/X_test_metadata_dc.csv")
y_train <- train_metadata$dc_detailed
y_train <- factor(y_train)

y_test <- test_metadata$dc_detailed
y_test <- factor(y_test, levels = levels(y_train))

X_train_topics <- read_csv("data/X_train_topics_dc.csv")
colnames(X_train_topics) <- topic_labels
X_test_topics <- read_csv("data/X_test_topics_dc.csv")
colnames(X_test_topics) <- topic_labels
X_train_style <- read_csv("data/X_train_style_dc.csv")
colnames(X_train_style) <- style_labels
X_test_style <- read_csv("data/X_test_style_dc.csv")
colnames(X_test_style) <- style_labels

X_train <- bind_cols(X_train_topics, X_train_style)
X_test <- bind_cols(X_test_topics, X_test_style)

train <- bind_cols(X_train, genre = y_train)
test <- bind_cols(X_test, genre = y_test)

glimpse(train)
glimpse(test)
levels(train$genre) == levels(test$genre)

model <- rule_fit(
  mode = "classification",
  trees = 100,
  mtry = 19,
  min_n = 4,
  tree_depth = 3,
  learn_rate = 0.01
) |>
  set_engine("xrf", seed = 42) |>
  set_mode("classification")
fitted_model <- fit(model, genre ~ ., data = train, verbose = TRUE)
# takes very long (at least on my computer)

y_pred <- predict(fitted_model, test)$.pred_class
