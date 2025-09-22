cat > diag_libs.R <<'R'
libs <- c("quantmod","tidyverse","caret","lubridate","xts","zoo","TTR","keras","reticulate")
cat("Starting library check\n")
for (lib in libs) {
  cat("Loading:", lib, " ... ")
  res <- tryCatch({
    library(lib, character.only = TRUE)
    cat("OK\n")
    TRUE
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    FALSE
  }, warning = function(w) {
    cat("WARNING:", conditionMessage(w), "\n")
    TRUE
  })
  if (!res) {
    cat("FAILED loading", lib, "\n")
    quit(status = 1)
  }
}
cat("All libraries loaded\n")
R
Rscript --vanilla diag_libs.R > diag_output.txt 2>&1 || true
type diag_output.txt
