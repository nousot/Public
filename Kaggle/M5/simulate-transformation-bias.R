library(ggplot2)


set.seed(543)
#set.seed(414593)
T <- 100
mu <- 20 #.2 # 20 

sim_df <- data.frame(t=1:T, y=rpois(T, mu), type='data')


# Loess on the original scale
mod_orig <- loess(y ~ t, data=sim_df)
sim_df_loess <- sim_df
sim_df_loess$y <- predict(mod_orig)
sim_df_loess$type <- 'loess orig scale'


# Loess on the sqrt scale
mod_tr <- loess(sqrt(y) ~ t, data=sim_df)
sim_df_tr_loess <- sim_df
sim_df_tr_loess$y <- predict(mod_tr) ^ 2
sim_df_tr_loess$type <- 'loess retransformed'


plot_df <- rbind(sim_df, sim_df_loess, sim_df_tr_loess)


p <- ggplot(plot_df, aes(x=t, y=y, color=type)) +
    geom_point() +
    ggtitle('Two loess fits using the same count data. What gives?')
p
