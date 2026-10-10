pkgs <- c("randomForest","e1071","glmnet","gbm","xgboost","MASS","nnet",
          "ggplot2","dplyr","tidyr","readr","scales","viridis",
          "shiny","plotly","DT","PMCMRplus","rpart")
ip <- rownames(installed.packages())
for (p in pkgs) cat(sprintf("  %-20s %s\n", p, ifelse(p %in% ip, "OK", "MISSING")))
