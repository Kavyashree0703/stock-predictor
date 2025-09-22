# data_prep.R
library(quantmod)
library(tidyverse)
library(caret)
library(lubridate)

# Parameters
symbol <- "AAPL"
window_size <- 60
train_end_date <- Sys.Date() - 30  # keep last 30 days for potential test/validation

# Fetch data
getSymbols(symbol, src = "yahoo", auto.assign = TRUE)
df <- get(symbol) %>% 
  as_tibble(rownames = "date") %>% 
  mutate(date = as.Date(date)) %>%
  rename(Open = 2, High = 3, Low = 4, Close = 5, Volume = 6, Adjusted = 7) %>%
  select(date, Close) %>%
  arrange(date)

# Ensure enough data
if (nrow(df) < (window_size + 10)) stop("Not enough historical data")

# Split into training portion (you can expand to train/val/test)
train_df <- df %>% filter(date <= train_end_date)

# Use Close only (you can add features later)
prices <- train_df$Close %>% as.numeric() %>% matrix(ncol=1)

# Fit a Min-Max scaler using caret preProcess
scaler <- preProcess(prices, method = c("range"))   # 0-1 scaling
prices_scaled <- predict(scaler, prices) %>% as.numeric()

# Build sequences (X = 60 previous closes, y = next day close)
create_sequences <- function(series, window) {
  X <- list()
  y <- c()
  for (i in seq_len(length(series) - window)) {
    X[[i]] <- series[i:(i + window - 1)]
    y[i] <- series[i + window]
  }
  X_arr <- array(unlist(X), dim = c(length(X), window, 1))
  list(X = X_arr, y = matrix(y, ncol=1))
}

seqs <- create_sequences(prices_scaled, window_size)
X_train <- seqs$X
y_train <- seqs$y

# Save objects
saveRDS(scaler, file = "scaler.rds")
saveRDS(list(X = X_train, y = y_train), file = "train_data.rds")

cat("Saved scaler.rds and train_data.rds. X shape:", dim(X_train), "y shape:", dim(y_train), "\n")
