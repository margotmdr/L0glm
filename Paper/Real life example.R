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



# Wisconsin breast cancer dataset
# cf https://github.com/Microsoft/microsoft-r/tree/master/microsoft-ml/Samples/101/BinaryClassification/BreastCancerPrediction
# https://medium.com/@kathrynklarich/exploring-and-evaluating-ml-algorithms-with-the-wisconsin-breast-cancer-dataset-506194ed5a6a
# https://rpubs.com/elena_petrova/breastcancer
# https://www.kaggle.com/lbronchal/breast-cancer-dataset-analysis
# https://www.kaggle.com/vincentlugat/breast-cancer-analysis-and-prediction
# https://www.kaggle.com/mirichoi0218/classification-breast-cancer-or-not-with-15-ml
# https://www.kaggle.com/gpreda/breast-cancer-prediction-from-cytopathology-data
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/kernels?sortBy=hotness&group=everyone&pageSize=20&datasetId=180&language=R
# https://www.kaggle.com/mercheovejero/breast-cancer-analysis-real-machine-learning
# https://www.kaggle.com/stevensmiley/ml-for-diagnosing-breast-cancer-steven-smiley


library(mlbench) # has data(BreastCancer)
library(caret)
library(dplyr)
library(MicrosoftML)

library(glmulti)
library(afex)
library(ROCR)

set_sum_contrasts()

data(BreastCancer)
BreastCancer$Label[BreastCancer$Class == "benign"] <- 0
BreastCancer$Label[BreastCancer$Class == "malignant"] <- 1
breastCancerDS <- select(BreastCancer, -Id, -Class)
breastCancerDS <- breastCancerDS[complete.cases(breastCancerDS),]
n <- nrow(breastCancerDS)
set.seed(1234)
train.ind <- sample(1:n, size = round(0.8 * n))
val.ind <- -train.ind
# train.ind <- 1:n
# val.ind <- 1:n

# Perform L0 penalized regression using main effects + all 1st order covariate interactions
mf <- stats::model.frame(formula = Label ~ .^2, data = breastCancerDS)
mt <- attr(mf, "terms")
X <- model.matrix(mt, mf, contrasts)
X_train <- X[train.ind,]
X_val <- X[val.ind,]
dim(X) # 683 x 2925

library(L0glm)
L0glm.model <- L0glm(formula = Label ~ .^2, data = breastCancerDS,
                     family=binomial(link=logit),
                     lambda = 10^seq(-1,1,length.out = 50),
                     tune.meth = "trainval",
                     train.vs.val = 0.8,
                     tune.crit = "class")
plot(x = L0glm.model$lambda.tune$lambdas, y = L0glm.model$lambda.tune$IC[,"class"], log = "xy", type = "l")
abline(v = L0glm.model$lambda.tune$best.lam)
L0glm.model$lambda.tune$best.lam # 3.55648
matplot(x = L0glm.model$lambda.tune$lambdas, y = L0glm.model$lambda.tune$coefficients.lam, log = "x", type = "l")
abline(v = L0glm.model$lambda.tune$best.lam)

# Predict validation set with L0glm
L0glm.preds <- binomial()$linkinv(X %*% L0glm.model$coefficients)
# L0glm.preds <- L0glm.model$fitted.values
sum(L0glm.model$coefficients!=0) # 3 !
colnames(model.matrix(Label ~ .^2, data = breastCancerDS))[which(L0glm.model$coefficients!=0)] # 1, 2, 47
L0glm.model$coefficients[L0glm.model$coefficients!=0]
#    (Intercept) Cl.thickness.L   Bare.nuclei1
#    0.8773344      6.9241845     -3.2228664

perf.L0glm <- performance(prediction(L0glm.preds[val.ind], breastCancerDS$Label[val.ind]),"tpr","fpr")
perf.L0glm <- performance(prediction(L0glm.preds, breastCancerDS$Label),"tpr","fpr")
plot(perf.L0glm, colorize=FALSE, col = "red3", main = "ROC curve on validation set")
abline(a = 0, b = 1, lty = 1, col = "grey40")

auc = performance(prediction(L0glm.preds[val.ind], breastCancerDS$Label[val.ind]), measure = "auc")
auc = performance(prediction(L0glm.preds, breastCancerDS$Label), measure = "auc")
auc = auc@y.values[[1]]
auc # 0.9911765 using val data only, 0.9817 when using all data
