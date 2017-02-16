#!/usr/bin/Rscript

args <- commandArgs(trailingOnly = TRUE)
r <- read.table(args[1])
r <- as.matrix(r)

print(svd(r)$v)
q()
