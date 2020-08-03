library(ggplot2)

set.seed(543)

# Generate Poisson data ------------------------------------------------------
T <- 100
mu <- .20  # Change this from .20 to 20 
sim_df <- data.frame(t=1:T, y=rpois(T, mu), type='Y_t')


# Local Mean (loess) model on the original scale
mod_orig <- loess(y ~ t, data=sim_df)
sim_df_loess <- sim_df
sim_df_loess$y <- predict(mod_orig)
sim_df_loess$type <- 'hat E(Y_t)'


# Local Mean (loess) model on the sqrt scale
mod_tr <- loess(sqrt(y) ~ t, data=sim_df)
sim_df_tr_loess <- sim_df
sim_df_tr_loess$y <- predict(mod_tr) ^ 2
sim_df_tr_loess$type <- 'hat E(sqrt(Y_t)) ^ 2'


# Analysis of bias on the square root scale
expected_bias <- -1 / (8 * sqrt(mu)) + 1 / (16 * mu ^ (1.5))
observed_bias <- mean(sqrt(sim_df_tr_loess$y) - sqrt(sim_df_loess$y))
cat("On the square root scale, a Taylor expansion estimates a bias of",
    expected_bias, "\nwhereas the mean difference in loess estimates",
    "(on the sqrt scale) is", observed_bias, "\n")

plot_df <- rbind(sim_df, sim_df_loess, sim_df_tr_loess)

p <- ggplot(plot_df, aes(x=t, y=y, color=type)) +
  geom_point() +
  xlab("time (t), the sequential order of observed data") +
  ylab(paste("Scale of Poisson data (Y_t) with mean", mu)) +
  ggtitle('Loess estimation of the mean under transformation scenario')
p
p + ggsave(paste0('C:/devl/jenson-demo-mu-is-', mu, '.png'), width = 7, height = 7)
