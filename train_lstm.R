# train_lstm.R
library(keras)
library(tensorflow)

# Load training data
train <- readRDS("train_data.rds")
X_train <- train$X
y_train <- train$y

# Model architecture
timesteps <- dim(X_train)[2]
features <- dim(X_train)[3]

model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(timesteps, features), return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 32) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = list("mean_absolute_error")
)

summary(model)

# Fit model
history <- model %>% fit(
  X_train, y_train,
  epochs = 30,
  batch_size = 32,
  validation_split = 0.1,
  callbacks = list(callback_early_stopping(monitor="val_loss", patience=6, restore_best_weights = TRUE))
)

# Save model
save_model_hdf5(model, "stock_lstm.h5")
cat("Model saved to stock_lstm.h5\n")
