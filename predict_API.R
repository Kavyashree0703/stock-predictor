# predict_api.R
library(plumber)
library(quantmod)
library(keras)

#* @apiTitle Stock Predictor API

# load model and scaler at startup
model <- load_model_hdf5("stock_lstm.h5")
scaler <- readRDS("scaler.rds")
window_size <- 60

predict_from_symbol <- function(symbol) {
  # fetch last 100 days to be safe
  tmp <- tryCatch({
    getSymbols(symbol, src = "yahoo", auto.assign = FALSE)
  }, error = function(e) NULL)
  if (is.null(tmp)) stop("Could not fetch symbol")
  
  df <- as.data.frame(tmp)
  close_prices <- df[, "AAPL.Close"]
  # but column name depends on symbol, so more robust:
  cp_col <- grep("Close$", colnames(df), value = TRUE)
  close_prices <- df[, cp_col]
  close_prices <- as.numeric(tail(close_prices, 200))  # last 200 rows max
  
  if (length(close_prices) < window_size) stop("Not enough data for prediction")
  
  # scale using saved scaler (caret -> preProcess) -> predict expects a data.frame/matrix
  scaled <- predict(scaler, as.data.frame(matrix(close_prices, ncol=1)))
  scaled_vec <- as.numeric(tail(scaled[,1], window_size))
  X_input <- array(scaled_vec, dim = c(1, window_size, 1))
  
  pred_scaled <- predict(model, X_input)
  # inverse transform: caret preProcess range transforms with stored range
  # To invert: x_orig = (x_scaled * (max - min)) + min
  mins <- scaler$ranges["min", 1]
  maxs <- scaler$ranges["max", 1]
  pred_orig <- as.numeric(pred_scaled) * (maxs - mins) + mins
  
  list(symbol = toupper(symbol), prediction = round(pred_orig, 2))
}

#* Predict stock price
#* @param symbol Stock ticker symbol
#* @get /predict
function(symbol = "AAPL") {
  tryCatch({
    res <- predict_from_symbol(symbol)
    res
  }, error = function(e) {
    list(error = e$message)
  })
}
