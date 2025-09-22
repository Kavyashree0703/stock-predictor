# shiny_app.R
library(shiny)
library(plotly)
library(quantmod)
library(keras)

model <- load_model_hdf5("stock_lstm.h5")
scaler <- readRDS("scaler.rds")
window_size <- 60

ui <- fluidPage(
  titlePanel("Stock Predictor (R / Keras)"),
  sidebarLayout(
    sidebarPanel(
      textInput("symbol", "Ticker", value = "AAPL"),
      actionButton("pred", "Predict"),
      br(),
      verbatimTextOutput("pred_text")
    ),
    mainPanel(
      plotlyOutput("price_plot")
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$pred, {
    symbol <- toupper(input$symbol)
    tmp <- tryCatch(getSymbols(symbol, src = "yahoo", auto.assign = FALSE), error = function(e) NULL)
    if (is.null(tmp)) {
      output$pred_text <- renderText("Unable to fetch data for symbol")
      return()
    }
    df <- as.data.frame(tmp)
    cp_col <- grep("Close$", colnames(df), value = TRUE)
    closes <- as.numeric(tail(df[, cp_col], 200))
    if (length(closes) < window_size) {
      output$pred_text <- renderText("Not enough history for prediction")
      return()
    }
    # prepare scaled input
    scaled <- predict(scaler, as.data.frame(matrix(closes, ncol=1)))
    scaled_vec <- as.numeric(tail(scaled[,1], window_size))
    X_input <- array(scaled_vec, dim = c(1, window_size, 1))
    pred_scaled <- predict(model, X_input)
    mins <- scaler$ranges["min", 1]
    maxs <- scaler$ranges["max", 1]
    pred_orig <- as.numeric(pred_scaled) * (maxs - mins) + mins
    
    output$pred_text <- renderText(paste0(symbol, " predicted next close: $", round(pred_orig, 2)))
    
    # plot last 60 closes + predicted point
    hist_closes <- tail(closes, window_size)
    xvals <- seq_along(hist_closes)
    p <- plot_ly(x = xvals, y = hist_closes, type = 'scatter', mode = 'lines+markers', name = 'Historical Close') %>%
      add_markers(x = max(xvals) + 1, y = pred_orig, name = 'Predicted', marker = list(color = 'red', size = 10)) %>%
      layout(title = paste(symbol, " - last 60 closes + predicted"), xaxis = list(title = "Index"), yaxis = list(title = "Price"))
    output$price_plot <- renderPlotly(p)
  })
}

shinyApp(ui, server)
