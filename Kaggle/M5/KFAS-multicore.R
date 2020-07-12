library(doParallel)
library(foreach)
library(dplyr)
library(readr)
library(tidyr)
library(KFAS)

df <- read_csv('data/M5_lean.csv.gz')
split_df <- split(df, paste(df$item_id, df$store_id, '_'))
rm(df)
gc()

cl <- parallel::makeCluster(31)
registerDoParallel(cl)


do_kfas <- function(i) { 
  one_store <- split_df[[i]]
  
  p_NA <- sum(is.na(one_store$sales)) / nrow(one_store)
  if (p_NA > .90) {
    # The NAs from the gaps become 0s if there are only NAs and 1s
    print("filling in NAs with 0s")
    one_store$sales <- ifelse(is.na(one_store$sales), 0, one_store$sales)
  }
  # Transformation
  one_store$sales2 <- one_store$sales # sqrt(one_store$sales)
  m <- mean(one_store$sales2, na.rm=T)
  s <- sd(one_store$sales2, na.rm=T)
  one_store$sales2 <- (one_store$sales2 - m) / s
  mod <- SSModel(sales2 ~
      	   SSMtrend(degree=2, Q=list(matrix(NA), matrix(NA)))
  	   + SSMcycle(365.25, Q=0, a1=c(.01, .01), P1=diag(c(.01, .01)))
  	   + SSMseasonal(7, Q=0, sea.type="dummy")
  	   + SSMregression(~ sell_price, Q=0, a1=0, P1=4, remove.intercept=TRUE)
  	   + SSMregression(~ snap, Q=0, a1=0, P1=4, remove.intercept=TRUE)
             , H = matrix(NA), data = one_store)

  obj <- function(parms) {
    mod["T"][4, 4, 1] <- plogis(parms[1])
    mod["Q"][3, 3, 1] <- exp(parms[2])
    mod["Q"][4, 4, 1] <- exp(parms[3]) 
    mod["H"][1, 1, 1] <- exp(parms[4])

    kfs <- KFS(mod)
    posterior <-  -(kfs$logLik
        + dbeta(plogis(parms[1]), 50, 1, log=TRUE)
        + dexp(exp(parms[2]), 10, log=TRUE)
        + dexp(exp(parms[3]), 20, log=TRUE)
       )
    posterior
  }
  opt <- optim(c(3, -11, -13, 1.5), obj)

  parms <- opt$par
  mod["T"][4, 4, 1] <- plogis(parms[1])
  mod["Q"][3, 3, 1] <- exp(parms[2])
  mod["Q"][4, 4, 1] <- exp(parms[3]) 
  mod["H"][1, 1, 1] <- exp(parms[4])
  
  filtered <- KFS(mod)     
  
  one_store$pred_z <- as.numeric(filtered$muhat[, 1])
  one_store$sales_hat <- s * one_store$pred_z + m

  one_store$sales_hat <- ifelse(one_store$sales_hat < 0, 0, one_store$sales_hat) 

  # Creating submission version
  future_df <- one_store %>%
    filter(one_store$t > 1913) %>%
    mutate(F = paste0('F', ((t - 1914) %% 28) + 1),
           id = paste(item_id, store_id,
		    ifelse(t > 1941, 'evaluation', 'validation'), sep="_"))
  wide_future <- pivot_wider(future_df, id_cols = 'id', names_from="F",
                             values_from="sales_hat")
  wide_future
}



t1 <- proc.time()
results_df1 <- foreach(i=1:10000,
               .combine=rbind,
               .packages=c("dplyr", "tidyr", "KFAS")) %dopar%
    do_kfas(i)
ds <- proc.time() - t1

results_df2 <- foreach(i=10001:20000,
               .combine=rbind,
               .packages=c("dplyr", "tidyr", "KFAS")) %dopar%
    do_kfas(i)


results_df3 <- foreach(i=20001:length(split_df),
               .combine=rbind,
               .packages=c("dplyr", "tidyr", "KFAS")) %dopar%
    do_kfas(i)


results_df <- rbind(results_df1, results_df2, results_df3)

write_csv(results_df1, paste0('submission1of3', gsub(" |:", "", date()), '.csv'))
