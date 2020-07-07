path2multgam <- "your_path_to_the_location_where_multgam_has_been_extracted"

library(Rcpp)

compileAttributes(path2multgam)
install.packages(path2multgam, repos=NULL, type="source")
