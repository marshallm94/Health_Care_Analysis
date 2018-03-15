library(pwr)
df = read.csv("/Users/marsh/galvanize/dsi/projects/health_capstone/anova.csv")
names(df) <- c('X','state','year','percent_uninsured')
df$year <- as.factor(df$year)
year_anova <- aov(percent_uninsured ~ year, data = df)
state_anova <- aov(percent_uninsured ~ state, data=df)
summary(year_anova)
summary(state_anova)
pwr.anova.test(k = 9, n = 3192, f=0.05, sig.level=.05)
p_vals <- TukeyHSD(year_anova)$year[,4]
significant_p_vals <- p_vals <= 0.05
p_val_df <- as.data.frame(p_vals[significant_p_vals])
p_val_df$year_combo <- rownames(p_val_df)
p_val_df[order(p_val_df$`p_vals[significant_p_vals]`),]

