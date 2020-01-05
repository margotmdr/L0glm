library(export)

sim <- simulate_spike_train()
X <- sim$X
y <- sim$y

matplot(X[1:50,3:48], type = "l", col = "red3", lty = 1:2, xlab = "Index", ylab = "")
graph2ppt(file = "D:/Documents/Dropbox/christophe/PPTs/190809 - lab presentation - files/Figures.pptx", append = T)

plot(sim$a[1:50], col = "red3", type = "h", xlab = "Index", ylab = "")
graph2ppt(file = "D:/Documents/Dropbox/christophe/PPTs/190809 - lab presentation - files/Figures.pptx", append = T)

plot(sim$y[1:50], col = "grey30", pch = 16, xlab = "Index", ylab = "Response")
graph2ppt(file = "D:/Documents/Dropbox/christophe/PPTs/190809 - lab presentation - files/Figures.pptx", append = T)

L0glm.out <- L0glm(formula = "y ~ 0 + .", data = data.frame(X, y = y),
                   family = poisson(identity),
                   lambda = 1,
                   nonnegative = TRUE, normalize = FALSE)
plot_benchmark(x = sim$x, y = y, fit = L0glm.out, a.true = sim$a,
                     main="Ground truth vs L0 penalized L0glm estimates")
graph2ppt(file = "D:/Documents/Dropbox/christophe/PPTs/190809 - lab presentation - files/Figures.pptx", append = T)



# Ravoet et al. 2013

library(glmulti)
library(afex)
library(ROCR)

set_sum_contrasts()

bee.df <- read.csv("D:/Documents/Dropbox/christophe/PPTs/190809 - lab presentation - files/beepathogens.csv")
bee.df <- bee.df[,!colnames(bee.df) %in% c(("spiroplasma_apis"))] # Remove spiroplasma_apis because no entropy
bee.df <- bee.df[-which(is.na(bee.df), arr.ind = TRUE)[,1],]
n <- nrow(bee.df)
set.seed(1234)
train.ind <- sample(1:nrow(bee.df), size = round(0.8 * nrow(bee.df)))
val.ind <- -train.ind
train.ind <- 1:n
val.ind <- 1:n

# Perform exhaustive search on limited covariates
models <- glmulti(died ~ nosema_ceranae + varroa + crithidia + dwv + nr_pathogens,
                  data = bee.df[train.ind,],
                  family=binomial(link=logit),
                  level = 2, crit = "aic",
                  plotty=T, report=T, method="h")
bestmodel <- glm(died~1+nosema_ceranae+varroa+crithidia+nr_pathogens+crithidia:nosema_ceranae+varroa:nr_pathogens,family=binomial(link=logit),data=bee.df[train.ind,])
bestmodel <- glm(died~1+varroa+crithidia,family=binomial(link=logit),data=bee.df[train.ind,])

# Perform L0 penalized on all possible 2-level covariate interactions
L0glm.model <- L0glm(died ~ .^2, data = bee.df[train.ind,],
                     family=binomial(link=logit),
                     lambda = 10^seq(-1,1,length.out = 50),
                     tune.meth = "3-fold",
                     tune.crit = "class")
# plot(x = L0glm.model$lambda.tune$lambdas, y = L0glm.model$lambda.tune$IC[,"class"], log = "xy", type = "l")
# matplot(x = L0glm.model$lambda.tune$lambdas, y = L0glm.model$lambda.tune$coefficients.lam, log = "x", type = "l")
# abline(v = L0glm.model$lambda.tune$best.lam)

# Predict validation set with glm
glm.preds <- predict(bestmodel, type = "response", newdata = bee.df[val.ind,])

# Get the validation design matrix for L0glm
mf <- stats::model.frame(formula = died ~ .^2, data = bee.df[val.ind,], drop.unused.levels = TRUE)
mt <- attr(mf, "terms")
X <- model.matrix(mt, mf, contrasts)
# Predict validation set with L0glm
L0glm.preds <- 1 - binomial()$linkinv(X %*% L0glm.model$coefficients)
# L0glm.preds <- L0glm.model$fitted.values

perf.glm <- performance(prediction(glm.preds, bee.df$died[val.ind]),"tpr","fpr")
perf.L0glm <- performance(prediction(L0glm.preds, bee.df$died[val.ind]),"tpr","fpr")
plot(perf.glm,colorize=FALSE, col = "red3", main = "ROC curve on validation set")
lines(perf.L0glm@x.values[[1]], perf.L0glm@y.values[[1]], col = "green4")
abline(a = 0, b = 1, lty = 1, col = "grey40")
graph2ppt(file = "D:/Documents/Dropbox/christophe/PPTs/190809 - lab presentation - files/Figures.pptx", append = T)


