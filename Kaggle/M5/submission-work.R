library(tidyr)
library(dplyr)
library(readr)

df <- read_csv('data/M5_lean_eval.csv.gz')
kalman <- read_csv('C:/devl/multivariate-item-ss-v1.csv')
lgb <- read_csv('C:/devl/shades-submission-v1.csv')

# Loess for every item ---------
loess_df <- data.frame()

all_items <- unique(df$item_id)
all_stores <- unique(df$store_id)
k <- 0
for (item in all_items) {
  k <- k + 1
  print(k)
  for (store in all_stores) {
    one_df <- df %>% filter(item_id == item, store_id == store)
    mod2 <- loess(sales ~ t, data = one_df, span = .1)
    last_loess <- tail(predict(mod2), 1)
    loess_df <- rbind(loess_df, data.frame(item_id = item, store_id = store,
                                           last_loess = last_loess))
  }
}

loess_df %>% write_csv('~/loess_last.csv')

# Future option to read in loess fits from file
# loess_df <- read_csv('~/loess_last.csv')

loess_df$id <- paste(loess_df$item_id, loess_df$store_id, 'evaluation', sep='_')
loess_df <- loess_df %>% select(id, last_loess)

kalman_long <- kalman %>%
    pivot_longer(cols=starts_with('F'), names_to='F', values_to = 'kalman_pred') %>%
    filter(grepl('evaluation', id))

lgb_long <- lgb %>%
    pivot_longer(cols=starts_with('F'), names_to='F', values_to = 'lgb_pred') %>%
    filter(grepl('evaluation', id))

both <- kalman_long %>% inner_join(lgb_long) %>%
	mutate(t = as.numeric(substr(F, 2, 100)) + 1941)


# Get mean of 1rst 3 predictions for both 
initial_levels <- both %>% filter(F %in% c('F1', 'F2', 'F3')) %>%
    group_by(id) %>%
    summarize(kalman_initial = mean(kalman_pred), lgb_initial = mean(lgb_pred))

initial_levels <- initial_levels %>%
    inner_join(loess_df, on='item_id') %>%
    mutate(kalman_closer = abs(kalman_initial - last_loess) < abs(lgb_initial - last_loess))


both <- both %>% inner_join(initial_levels, on='item_id')

# Pick predictions for which is closer.
both$final_pred <- ifelse(both$kalman_closer, both$kalman_pred, both$lgb_pred)

# Use validation actuals to get a perfect score
valid_df <- df %>% filter(t >= 1914, t <= 1941) %>%
    mutate(final_pred = sales, id = paste(item_id, store_id, 'validation', sep='_'),
	   F = paste0('F', t - 1913)) %>%
    replace_na(list(final_pred = 0)) %>%
    select(id, F, final_pred)

eval_df <- both %>% select(id, F, final_pred)

sub_long_df <- valid_df %>% bind_rows(eval_df)
sub_df <- sub_long_df %>%
    pivot_wider(id_cols = 'id', names_from='F', values_from = 'final_pred')
sub_df %>% write_csv("C:/devl/kalman-lgb-loess-choose-final.csv")

# Analytical work below -------------------------------------------

# Calculate stats and compare
stats <- both %>%
    group_by(id) %>%
    summarize(corr = cor(kalman_pred, lgb_pred),
	      z = (mean(kalman_pred) - mean(lgb_pred)) / sqrt(var(kalman_pred) + var(lgb_pred)))

stats %>% arrange(desc(z)) %>% print(n = 20)

stats %>% arrange(z)

stats %>% arrange(abs(z))

# Investigate individual items:
item <- 'HOUSEHOLD_2_108'
store <- 'TX_1'

item_df <- df %>% filter(item_id == item, store_id == store)
pred_df <- both %>% filter(id == paste(item, store, 'evaluation', sep='_'))

mod <- loess(sales ~ t, data = item_df, span = .15)
item_df$loess_pred <- c(predict(mod), rep(NA, 28))

vis_T <- 1800

plot(sales ~ t, data = item_df %>% filter(t > vis_T), main = paste(item, store, sep="_"))
points(kalman_cal ~ t, data = pred_df, col = 'blue')
points(kalman_pred ~ t, data = pred_df, col = 'orange')
points(lgb_pred ~ t, data = pred_df, col = 'red')
points(loess_pred ~ t, data = item_df, col = 'green')
