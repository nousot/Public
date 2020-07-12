library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(KFAS)

# Read in data
df <- read_csv('data/M5_lean_eval.csv.gz')

# Helper functions ----------------------------------------
all_items <- unique(df$item_id)

get_random_item <- function() {
    item <<- sample(all_items, 1)
    print(item)
}

# Clip future price function
clip_future_prices <- function(price_tb, train_t) {
  # Get maximums from pre_train_T period, clip future values
  cols <- setdiff(names(price_tb), 't')
  pre_train <- price_tb %>% filter(t <= train_t)
  
  for(col in cols) {
    max_val <- max(pre_train[, col], na.rm=T)
    future_vec <- price_tb[price_tb$t > train_t, col] %>% pull()
    if (max(future_vec, na.rm = T) > max_val) {cat('price clipping', col, '\n')}
    price_tb[price_tb$t > train_t, col] <- ifelse(future_vec > max_val, max_val, future_vec)

    min_val <- min(pre_train[, col], na.rm=T)
    future_vec <- price_tb[price_tb$t > train_t, col] %>% pull()
    if (min(future_vec, na.rm = T) < min_val) {cat('price clipping', col, '\n')}
    price_tb[price_tb$t > train_t, col] <- ifelse(future_vec < min_val, min_val, future_vec)

  }
  price_tb
}

scale_matrix <- function(mat) {
  factors <- apply(mat, 2, function(x) max(1, quantile(x, .95, na.rm=T)))
  no_na_mat <- mat
  no_na_mat[is.na(mat)] <- 0
  scaled <- no_na_mat %*% diag(1 / factors)
  scaled[is.na(mat)] <- NA
  list(scaled = scaled, factors = factors)
}

rescale_matrix <- function(mat_sc, factors) {
  no_na_mat <- mat_sc
  no_na_mat[is.na(mat_sc)] <- 0
  rescaled <- no_na_mat %*% diag(factors)
  rescaled[is.na(mat_sc)] <- NA
  rescaled
}

# Poor man's tests
X_mat <- matrix(c(1, 4, 5, 2, 3, 5, NA, 2), ncol=2)
X_mat_sc <- scale_matrix(X_mat)
s1 <- quantile(X_mat[, 1], .95)
s2 <- quantile(X_mat[, 2], .95, na.rm=T)
X_mat_sc$scaled[2, 1] == 4 / s1
X_mat_sc$scaled[2, 2] == 5 / s2
X_mat_sc$factors[1] == s1
X_mat_sc$factors[2] == s2

X_mat2 <- rescale_matrix(X_mat_sc$scaled, X_mat_sc$factors)
round(X_mat2, 6) == X_mat
sum(is.na(X_mat2[, 1])) == 0
is.na(X_mat2[3, 2])

# Clip future price test
# t = 5 is the cutoff for training
price_matrix <- matrix(c(1,2,3,4,5,6,7,
                         1,3,NA,1,2,19,NA,
                         1,3,2,1,2,1.5, 0.1), ncol=3)
prices <- tibble(as.data.frame(price_matrix))

names(prices) <- c('t', 'CA_1', 'CA_2')

clipped <- clip_future_prices(prices, train_t = 5)
clipped5 <- clipped %>% filter(t == 5)
clipped6 <- clipped %>% filter(t == 6)
clipped7 <- clipped %>% filter(t == 7)

clipped5$CA_1 == 2
clipped6$CA_1 == 3  # clipped to max
is.na(clipped7$CA_1)
clipped6$CA_2 == 1.5
clipped7$CA_2 == 1

# Multivariate ------------------------------------------------------

#TRAIN_T <- 1857
TRAIN_T <- 1941 # last row used in training
PHI <- .95
p <- 10 # 10 storeS
PLOT <- FALSE


results_df <- data.frame()
cv_df <- data.frame()
params_df <- data.frame()

# For storing these until we build variance in
factors_list <- list()
var_list <- list()

k <- 0

#loop_items <- sample(all_items, 2)
loop_items <- all_items
#for (PHI in c(.9, .95, .97)) {  # CV-Loop
#for (TRAIN_T in c(1885, 1913)) {

for (item in loop_items) {
  k <- k + 1
  cat('\n', k, item, '--------------------------------------\n')
 
  item_df <- df %>% filter(item_id == item) %>% arrange(t)
  first_sales_t <- item_df %>% filter(sales > 0) %>%
    summarize(first_t = min(t)) %>% pull()

  # Create a sales matrix with columns ["t", "CA_1", ..., "WI_3"]
  Y <- pivot_wider(item_df, id_cols="t", names_from="store_id",
                   values_from="sales")
  store_names <- tail(names(Y), 10)
 
  # Prepare Multivariate snap regressor
  Snap <-  pivot_wider(item_df, id_cols="t", names_from="store_id",
                       values_from="snap")
  Snap <- Snap %>% filter(t >= first_sales_t) %>%
    select(t, snap = CA_1)
 
  # Prepare Multivariate sell_price_regressor
  SP <-  pivot_wider(item_df, id_cols="t", names_from="store_id",
                    values_from="sell_price")
  SP <- clip_future_prices(SP, TRAIN_T)
  SP <- SP %>% fill(names(SP), .direction="up")
  SP_scaled <- scale_matrix(as.matrix(SP[2:11]))$scaled # don't need to scale back
  colnames(SP_scaled) <- store_names
  SP_scaled <- as.data.frame(cbind(1:1969, SP_scaled))
  names(SP_scaled)[1] <- 't'

  SP_long <- pivot_longer(SP_scaled, cols=all_of(store_names),
                          names_to="state_id",
  			  values_to="sell_price")
  SP_long <- SP_long %>% filter(t >= first_sales_t)
  SP_list <- split(SP_long, SP_long$state_id)

  train_Y <- as.matrix(Y[, 2:(p + 1)])
  train_Y[(TRAIN_T + 1):nrow(train_Y), ] <- NA
  train_Y <- train_Y[first_sales_t:nrow(train_Y), ]  

  # transformation of Y -------------
  train_Y <- sqrt(train_Y)
  
  # Scaling all endogenous series by the 95th percentile
  Y_scaled <- scale_matrix(train_Y)
  train_Y <- Y_scaled$scaled
  colnames(train_Y) <- store_names
  
  diag_na <- diag(rep(NA, p))

  mod <- SSModel(train_Y ~
  	   SSMtrend(degree=2, Q=list(diag_na, diag_na))
  	   + SSMcycle(365.25, a1=rep(0, 2 * p), P1=diag(rep(1, 2 * p)))
  	   + SSMseasonal(7, a1=rep(0, 6 * p), P1=diag(rep(1, 6 * p)), sea.type="dummy")
  	   + SSMregression(rep(list(~ sell_price), p), type = "common",
                           data = SP_list, remove.intercept = TRUE)
  	   + SSMregression(~ snap, type = "common",
                           data = Snap, remove.intercept = TRUE)
           , H = diag_na) # H is for gaussian

  # The dimensions of Q (if you specify 0, they appear
  rownames(mod$R)[rowSums(mod$R[, ,1])>0]
  offset <- 2 # terms placed above level and slope
  # It's level, slope for store 1, then level, slope for store 2
  mod$a1[seq(1 + offset, 19 + offset, 2)] <- .5 # assume starting in between 0 - 1 scale
  diag(mod$P1)[seq(1 + offset, 19 + offset, 2)] <- 1 # start slopes with a little regularization around 0
  diag(mod$P1)[seq(2 + offset, 20 + offset, 2)] <- .2 # start slopes with a little regularization around 0
  diag(mod$P1inf) <- 0

  mod$P1[1, 1] <- 1 # sell_price
  mod$P1[2, 2] <- 1  # snap

  obj <- function(params) {
    diag(mod["Q"][, , 1])[seq(1, 19, 2)] <- exp(params[1]) # common levels state var
    diag(mod["Q"][, , 1])[seq(2, 20, 2)] <- exp(params[2]) # common slopes state var
    diag(mod["H"][, , 1]) <- exp(params[3]) # common observation error

    -(logLik(mod)
     # Regularization
     + dexp(exp(params[1]), rate=10, log=TRUE)
     + dexp(exp(params[2]), rate=50, log=TRUE)
     )
  }
  opt <- optim(c(-7, -11, -1), obj, control=list(maxit=8), method='BFGS')
 
  # Using params in the model
  params <- opt$par
  diag(mod["Q"][, , 1])[seq(1, 19, 2)] <- exp(params[1])
  diag(mod["Q"][, , 1])[seq(2, 20, 2)] <- exp(params[2])
  diag(mod["H"][, , 1]) <- exp(params[3])

  # TODO: confirm offset with T
  diag(mod$T[ , , 1])[seq(offset + 2, offset + 20, 2)] <- PHI

  kfs <- KFS(mod)

  # Mean Matrix
  mean_mat <- kfs$muhat
  # Reconstitute
  rescaled_mean_mat <- rescale_matrix(mean_mat, Y_scaled$factors)
  colnames(rescaled_mean_mat) <- store_names

  # Storing Variance for later
  var_list[[item]] <- kfs$V_mu
  factors_list[[item]] <- Y_scaled$factors

  # dealing with sqrt
  rescaled_mean_mat[rescaled_mean_mat < 0] <- 0
  rescaled_mean_mat <- rescaled_mean_mat ** 2

  # Turn mean matrix into a data frame of the original dimensions 
  long_pred <- as.data.frame(rescaled_mean_mat) %>%
    pivot_longer(everything(), names_to='store_id', values_to='sales_hat')
  long_pred$t <- sort(rep(first_sales_t:1969, 10))

  # Merged in predictions with item_df
  item_df_aug <- item_df %>% inner_join(long_pred, by=c('store_id', 't')) 

  if (TRAIN_T < 1941) {
    item_cv <- item_df_aug %>%
        filter(t > TRAIN_T) %>%
        group_by(store_id) %>%
        summarize(mean_chi = mean((sales - sales_hat) ** 2 / ifelse(sales > 1, sales, 1), na.rm=T)) %>%
        mutate(phi = PHI, item_id = item) %>%
        as.data.frame()
    print(item_cv)
    cv_df <- cv_df %>% bind_rows(item_cv)
  }
  if (PLOT) {
      q <- item_df_aug %>%
         filter(t > 1900) %>%
         ggplot(aes(x=date, y=sales_hat, color=store_id)) +
         geom_point(aes(y=sales), color='black', alpha=.7) +
         geom_line(alpha = .8) +
         ggtitle(paste('Item:', item)) +
         ylab('Sales (actual and predicted)') + 
	 ylim(0, 20)
     print(q)
  }
  params_df <- params_df %>%
    bind_rows(data.frame(item_id = item, q_level = params[1],
                         q_slope = params[2], h = params[3], phi=PHI))
  results_df <- results_df %>% bind_rows(item_df_aug)
}

#} # END CVLOOP ----

# Create submission file ---
res <- results_df

future_df <- res %>%
  filter(res$t > 1913) %>%
  mutate(F = paste0('F', ((t - 1914) %% 28) + 1),
         id = paste(item_id, store_id,
		    ifelse(t > 1941, 'evaluation', 'validation'), sep="_"))

wide_future <- pivot_wider(future_df, id_cols = 'id', names_from="F",
                           values_from="sales_hat")
write_csv(wide_future, 'submissions/multivariate-item-ss-v1.csv')
save.image('~/big-image.RData')  # Wrote this line in w/out testing it
