####---- SCRIPT DESCRIPTION ----####


# This script shows how to use the package 'L0glm' and performs some test and
# benchmarks as comparison to other popular softwares and algorithms

####---- SETUP ENVIRONMENT ----####


setwd("~/GitHub/L0glm/Paper")
# devtools::install_github("tomwenseleers/L0glm")
library(L0glm)
library(l0ara) # used my fork with nonnegativity constraints here
# library(devtools)
# install_github("hazimehh/L0Learn", ref="nng") # L0Learn with nonnegativity constraints, https://github.com/hazimehh/L0Learn/issues/43
library(L0Learn)
library(nnls)
library(microbenchmark)
library(export)
library(ggplot2)
library(glmnet)
library(ncvreg)
# devtools::install_github("jaredhuling/ordinis")
library(ordinis)
library(bestsubset)


# graph2ppt(file = "Github/graphs") # Initialize the ppt file


####---- SHOWCASE L0GLM USING SIMULATED BLURRED SUPERIMPOSED SPIKE TRAIN WITH POISSON NOISE ----####


# Simulate some data
n <- 500
p <- 500
s <- 0.1 # sparsity as proportion of p that have nonzero coefficients
k <- round(p*s) # nr of nonzero covariates
sim <- simulate_spike_train(n = n, p = p, k = k,
                            mean_beta = 1000, sd_logbeta = 1,
                            family = "poisson", seed = 123, Plot = TRUE)


# Set up the parameters for controlling the algorithm
ctrl.fit <- control.fit.gen() # default
ctrl.iwls <- control.iwls.gen(maxit = 1)
ctrl.l0 <- control.l0.gen() # default

weighted.rmse <- function(actual, predicted, weight){
  sqrt(sum((predicted-actual)^2*weight)/sum(weight))
}
weightedrmse_betas <- function(betas) apply(betas, 2, function (fitted_coefs) {
  weighted.rmse(actual=sim$y_true,
                predicted=sim$X %*% fitted_coefs,
                weight=1/sim$y_true) } )

# Fit a GLM with Poisson error structure and identity link, with nonnegativity
# constraints on the coefficients, and L0 penalty with fixed lambda = 1
microbenchmark(L0glm_fit <- L0glm(y~0+., data = data.frame(y=sim$y, sim$X),
                   family = poisson(identity),
                   lambda = 1, nonnegative = TRUE, normalize = FALSE,
                   tune.meth = "none",
                   control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                   control.fit = ctrl.fit), times=1) # 24ms
# compare coefficient path of nnL0Learn, nnL0GLM & nnALASSO:
nnL0Learn_fit <- L0Learn.fit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                             loss = "SquaredError",
                             penalty="L0", algorithm="CDPSI",
                             maxSuppSize=round(p/2), intercept=FALSE, nLambda=100,
                             activeSetNum = 10, maxSwaps = 1000,
                             autoLambda=FALSE,
                             lambdaGrid=list(10^seq(-4,4,length.out=100)/ # DECREASING sequence recommended, but INCREASING works better under nonnegativity, see https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html
                                               (norm(sim$y*sqrt(1/(sim$y+1)),"2")^2)) )  # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
matplot(nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2),
        t(coef(nnL0Learn_fit)), type="l", lty=1, log="x")
wrmse_betas_L0Learn = weightedrmse_betas(coef(nnL0Learn_fit))
wrmse_betas_L0Learn[which.min(wrmse_betas_L0Learn)]
# 5.963164e-09 with increasing lambda seq (10^seq(-1,1,length.out=100)*normf)
# 5.87669e-09 with 10^seq(-2,0,length.out=100)
# 5.866426e-09 with 10^seq(-3,0,length.out=20)
# 5.657469e-09 with 10^seq(-4,1,length.out=20)
# 5.360266e-09 with 10^seq(-4,2,length.out=20)
# 5.338811e-09 with 10^seq(-4,3,length.out=100)
# 5.329658e-09 with 10^seq(-4,4,length.out=100)

# 6.532747e-09 with decreasing lambda seq (10^seq(1,-1,length.out=100)*normf)
# 6.281878e-09 with single fixed lambda value of 1
# 6.108157e-09 with single fixed lambda value of 1, replicated 20 times (10^seq(0,0,length.out=20))
beta_nnL0Learn <- as.vector(coef(nnL0Learn_fit,
                                 lambda=nnL0Learn_fit$lambda[[1]][which.min(wrmse_betas_L0Learn)],
                                 gamma=0))[1:p]
TP = sum(beta_nnL0Learn>0&sim$beta_true>0) # true positives
TP
# TP=45 with increasing lambda sequence
# TP=40 with decreasing lambda sequence
# TP=47 with lambda=1 replicated 20 times or with increasing seq of 10^seq(-4,3/4,length.out=100)
TN = sum(beta_nnL0Learn==00&sim$beta_true==0) # true negatives
TN
# TN=435 with increasing lambda sequence of 10^seq(-4,4,length.out=100)
# TN=434 with increasing lambda sequence of 10^seq(-4,3,length.out=100)
# TN=418 with increasing lambda sequence
# TN=415 with decreasing lambda sequence
# TN=408 with lambda=1 replicated 20 times
FP = sum(beta_nnL0Learn>0&sim$beta_true==0) # false positives
FP
# FP=15 with increasing lambda sequence (10^seq(-4,4,length.out=100))
# FP=16 with increasing lambda sequence (10^seq(-4,3,length.out=100))
# FP=18 with increasing lambda sequence (10^seq(-4,2,length.out=20))
# FP=23 with increasing lambda sequence (10^seq(-4,1,length.out=20))
# FP=35 with decreasing lambda sequence
# FP=42 with lambda=1 replicated 20 times
ACC = (TP+TN)/p # accuracy
ACC
# 0.964 with increasing lambda sequence (10^seq(-4,4,length.out=100))
# 0.962 with increasing lambda sequence (10^seq(-4,3,length.out=100))
# 0.958 with increasing lambda sequence (10^seq(-4,2,length.out=20))
# 0.948 with increasing lambda sequence (10^seq(-4,1,length.out=20))
# 0.91 wuth decreasing lambda sequence or with lambda=1 replicated 20 times

(nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2))[which.min(wrmse_betas_L0Learn)]
abline(v=(nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2))[which.min(wrmse_betas_L0Learn)], col="red")
beta_L0Learn_bestlambda = coef(nnL0Learn_fit)[,which.min(wrmse_betas_L0Learn)]
plot(nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2),
     wrmse_betas_L0Learn, type="l", log="xy")
abline(v=(nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2))[which.min(wrmse_betas_L0Learn)], col="red")

plot(sim$beta_true,beta_nnL0Learn,pch=16,col="steelblue")

lambda_L0Learn = (nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2))[which.min(wrmse_betas_L0Learn)]

# compare solution quality of some different popular sparse learners
microbenchmark("lmfit" = { lmfit <- lm.fit(x = sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)))
                           beta_lmfit <- lmfit$coefficients
                           beta_lmfit[is.na(beta_lmfit)] <- 0
                           beta_lmfit[beta_lmfit<0] <- 0 },
               "nnls" = { beta_nnls <- nnls(A = sim$X*sqrt(1/(sim$y+1)), b=sim$y*sqrt(1/(sim$y+1)))$x },
               "nnL0Learnfit" = { nnL0Learn_fit <- L0Learn.fit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                                            loss = "SquaredError",
                                                            penalty="L0", algorithm="CDPSI",
                                                            maxSuppSize=round(p/2), intercept=FALSE, nLambda=1,
                                                            activeSetNum = 10, maxSwaps = 1000,
                                                            autoLambda=FALSE,
                                                            lambdaGrid=list(lambda_L0Learn/(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2),1) )  # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
                                  beta_nnL0Learn <- as.vector(coef(nnL0Learn_fit))[1:p] },
               "doublennL0Learnfit" = { beta_doublennL0Learn <- rep(0,p)
                                        doublennL0Learn_fit <- L0Learn.fit(x=(sim$X*sqrt(1/(sim$y+1)))[,beta_nnL0Learn>0],
                                                                           y=sim$y*sqrt(1/(sim$y+1)),
                                                               loss = "SquaredError",
                                                               penalty="L0", algorithm="CDPSI",
                                                               maxSuppSize=round(p/2), intercept=FALSE, nLambda=1,
                                                               activeSetNum = 10, maxSwaps = 1000,
                                                               autoLambda=FALSE,
                                                               lambdaGrid=list(1/(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2)) )  # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
                                        beta_doublennL0Learn[beta_nnL0Learn>0] <- coef(doublennL0Learn_fit) },
               "nnl0arafit" = { nnl0ara_fit <- l0ara(x = sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                                family="gaussian", lam=1, standardize=FALSE, maxit=1E4, eps=1E-4)
                                beta_nnl0ara <- coef(nnl0ara_fit) },
               "L0glmfit_noblock" = { L0glm_fit <- L0glm.fit(X = sim$X, y = sim$y,
                                                      family = poisson(identity),
                                                      lambda = 1, nonnegative = FALSE, normalize = FALSE,
                                                      control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                                                      control.fit = ctrl.fit)
                                      beta_L0glm <- coef(L0glm_fit) },
               "nnL0glmfit_noblock" = { nnL0glm_fit <- L0glm.fit(X = sim$X, y = sim$y,
                                                   family = poisson(identity),
                                                   lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                                   control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                                                   control.fit = ctrl.fit)
                                        beta_nnL0glm <- coef(nnL0glm_fit) },
               "nnL0glmfit_block100" = { nnL0glm_fit_block100 <- L0glm.fit(X = sim$X, y = sim$y,
                                                      family = poisson(identity),
                                                      lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                                      control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                                                      control.fit = control.fit.gen(maxit=30, block.size = 100, tol=1E-3))
                                          beta_nnL0glm_block100 <- coef(nnL0glm_fit_block100) },
               "nnL0glmfit_L0Learn_prescreen" = { # nnL0Learn_fit <- L0Learn.fit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                                  #             loss = "SquaredError",
                                                  #             penalty="L0", algorithm="CD",
                                                  #             maxSuppSize=round(p/2), intercept=FALSE, nLambda=1,
                                                  #             autoLambda=FALSE,
                                                  #             lambdaGrid=list(1/(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2)) )  # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
                                                  # beta_nnL0Learn <- as.vector(coef(nnL0Learn_fit))
                                                   beta_nnL0glm_nnL0Learn <- rep(0,p)
                                                   nnL0glm_fit2 <- L0glm.fit(X = sim$X[,beta_nnL0Learn>0], y = sim$y,
                                                                            family = poisson(identity),
                                                                            lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                                                            control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                                                                            control.fit = ctrl.fit)
                                                   beta_nnL0glm_nnL0Learn[beta_nnL0Learn>0] <- coef(nnL0glm_fit2)
                                                   },
               "nnL0glmfit_nnls_prescreen" = {
                                    beta_nnL0glm_nnls <- rep(0,p)
                                    nnL0glm_fit3 <- L0glm.fit(X = sim$X[,beta_nnls>0], y = sim$y,
                                           family = poisson(identity),
                                           lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                           control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                                           control.fit = ctrl.fit)
                                    beta_nnL0glm_nnls[beta_nnls>0] <- coef(nnL0glm_fit3)
               },
               "nnLASSO_glmnet" = { lam.max <- function (X, y) max( abs(crossprod(X,y)) ) / nrow(X) # largest lambda value for LASSO so that no variables would be selected, cf https://stats.stackexchange.com/questions/166630/glmnet-compute-maximal-lambda-value & https://stats.stackexchange.com/questions/292147/how-to-find-the-smallest-lambda-such-that-all-lasso-elastic-net-coefficient?noredirect=1&lq=1
                                    lammax <- lam.max(X=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)))
                                    nnLASSO_cvfit <- cv.glmnet(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)), family="gaussian",
                                          lambda = exp(seq(log(lammax),-20,length.out=50)),
                                          intercept=F, standardize=F, nfolds=3, lower.limits=0)
                                    beta_nnLASSO_glmnet <- coef(nnLASSO_cvfit, s = nnLASSO_cvfit$lambda.1se)[-1] }, # lambda.1se or lambda.min give similar results
               "nnALASSO_ordinis" = { nnALASSO_fit <- ordinis(x=sim$X,
                                                              y=sim$y,
                                                              weights=1/(sim$y+1),
                                                              penalty = "alasso",
                                                              lower.limits = rep(0, p),
                                                              alpha = 1,
                                                              intercept = FALSE,
                                                              standardize = FALSE)
                                      beta_nnALASSO_ordinis <- nnALASSO_fit$beta[,which.min(BIC(nnALASSO_fit))][-1] },
               "nnMCP_ordinis" = { nnMCP_fit <- ordinis(x=sim$X,
                                                        y=sim$y,
                                                        weights=1/(sim$y+1),
                                                        penalty = "mcp",
                                                        lower.limits = rep(0, p),
                                                        alpha = 1,
                                                        intercept = FALSE,
                                                        standardize = FALSE)
                                    beta_nnMCP <- nnMCP_fit$beta[,which.min(BIC(nnMCP_fit))][-1] },
               "nnSCAD_ordinis" = { nnSCAD_fit <- ordinis(x=sim$X,
                                                        y=sim$y,
                                                        weights=1/(sim$y+1),
                                                        penalty = "scad",
                                                        lower.limits = rep(0, p),
                                                        alpha = 1,
                                                        intercept = FALSE,
                                                        standardize = FALSE)
                                     beta_nnSCAD <- nnSCAD_fit$beta[,which.min(BIC(nnSCAD_fit))][-1] },
               # "bestsubset" = { bestsubset_fit <- bs(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)), intercept=FALSE,
               #                               k = k) # set k to true value, since tuning it is super slow...
               #                  beta_bestsubset <- as.vector(coef(bestsubset_fit))
               #                  beta_bestsubset[beta_bestsubset<0] <- 0
               #                   },
               # often returns error Error 10020: Q matrix is not positive semi-definite (PSD), In lsfit(x[, ids[1:k]], y, int = FALSE) : 'X' matrix was collinea
               "relaxed_LASSO" = { relLASSO_fit <- lasso(x=sim$X*sqrt(1/(sim$y+1)),
                                         y=sim$y*sqrt(1/(sim$y+1)),
                                         intercept=FALSE, nrelax=5, nlam=50)
                                   betas_relLASSO <- coef(relLASSO_fit)
                                   betas_relLASSO[betas_relLASSO<0] <- 0
                                   beta_relLASSO <- betas_relLASSO[,which.min(weightedrmse_betas(betas_relLASSO))]
                                   },
               # "stepwise" = { stepwise_fit <- fs(x=sim$X*sqrt(1/(sim$y+1)),
               #                   y=sim$y*sqrt(1/(sim$y+1)), intercept=FALSE)
               # # returns error Error in `[<-`(`*tmp*`, A, k, value = backsolve(R, t(Q1) %*% y)) : subscript out of bounds
               #                betas_stepwise <- coef(stepwise_fit)
               #                beta_stepwise <- betas_stepwise[,which.min(weightedrmse_betas(betas_stepwise))] },
               times=1) #

betas <- data.frame(lmfit = beta_lmfit,
                    nnls = beta_nnls,
                    nnL0Learn = beta_nnL0Learn, # beta_L0Learn_bestlambda, # beta_nnL0Learn,
                    doublennL0Learn = beta_doublennL0Learn,
                    nnl0ara = beta_nnl0ara,
                    L0glm = beta_L0glm,
                    nnL0glm = beta_nnL0glm,
                    nnL0glm_block100 = beta_nnL0glm_block100,
                    nnL0glm_nnls = beta_nnL0glm_nnls,
                    nnL0glm_nnL0Learn = beta_nnL0glm_nnL0Learn,
                    nnLASSO_glmnet = beta_nnLASSO_glmnet,
                    nnALASSO_ordinis = beta_nnALASSO_ordinis,
                    nnMCP = beta_nnMCP,
                    nnSCAD = beta_nnSCAD,
                    # bestsubset = beta_bestsubset,
                    relLASSO = beta_relLASSO#,
                    # stepwise = beta_stepwise
                    )
thresh=1
betas[betas<thresh] <- 0
for (col in 1:ncol(betas)) {
plot(sim$beta_true, betas[,col], pch=16, col="steelblue", ylab="Fitted coefficients", xlab="True coefficients", main=colnames(betas)[col])
}

wrmse_betas = weightedrmse_betas(betas)
wrmse_betas
wrmse_betas[which.min(wrmse_betas)] # nnL0glm_nnL0Learn = 5.153607e-09
# see https://en.wikipedia.org/wiki/Sensitivity_and_specificity
FP <- colSums((betas>0)&(sim$beta_true==0))
FP
FP[FP==0]
TP <- colSums((betas>0)&(sim$beta_true>0))
TP[TP==k]
FN <- colSums((betas==0)&(sim$beta_true>0))
FN[FN==0]
TN <- colSums((betas==0)&(sim$beta_true==0))
TN[TN==(p-k)]
TPR = TP/(TP + FN) # sensitivity = recall = hit rate = true positive rate TPR = power = 1-FNR
TPR
TPR[which.max(TPR)] # nnls = 0.94
FPR = FP/(FP+TN) # false positive rate = fall-out = 1-TNR = Type I error rate
FPR
FPR[which.min(FPR)] # nnL0glm_nnL0Learn = 0.01333333
TNR = TN/(TN + FP) # specificity = selectivity = true negative rate TNR
TNR
TNR[which.max(TNR)] # nnL0glm_nnL0Learn = 0.9866667
TP/(TP+FP) # precision or positive predictive value
FP/(FP+TP) # false discovery rate FDR
FNR = FN/(FN+TP) # false negative rate FNR or miss rate = Type II error rate
FNR
FNR[which.min(FNR)] # nnls = 0.06
PLR = TPR / FPR # positive likelihood ratio (LR+)
PLR
PLR[which.max(PLR)] # nnL0glm_nnL0Learn = 69
NLR = FNR / TNR  # negative likelihood ratio (LR-)
NLR
NLR[which.min(NLR)] # nnL0Learn = 0.06221198
ACC = (TP+TN)/p # accuracy
ACC
ACC[which.max(ACC)] # nnL0glm_nnL0Learn = 0.98



# Perform inference on the coefficients. The function will automatically choose
# the correct inference procedure (non parametric bootstrapping in this case):
system.time(L0glm_infer_out <- L0glm.inference(L0glm_fit, level = 0.9, boot.repl = 1000,
                                   control.l0 = ctrl.l0, control.iwls = ctrl.iwls,
                                   control.fit = ctrl.fit, parallel = "snow",
                                   ncpus = 2)) # 8s

# Plot the results
plot_benchmark(x = sim$x, y = sim$y, fit = nnL0glm_fit, beta_true = sim$beta_true,
                     # inference = L0glm_infer_out,
                     main = "Estimated spike train (red=ground truth,\nblue/green=L0 penalized L0glm estimate, green=significant (1-sided p < 0.05)")
plot(sim$beta_true, L0glm_fit$coefficients, pch = 16, col="steelblue",
     main="L0GLM fitted vs expected coefficients", xlab="Expected true coefficients", ylab="Fitted coefficients", log="xy")

L0glm_infer_out$CI.lower
L0glm_infer_out$p.value # these p values don't look right... CIs frequently don't include 0 so should be significant
sum(L0glm_infer_out$p.value<0.05)
twosidep<-function(bootcoefs){ # func to calculate p values from bootstrap output
  mean( abs(bootcoefs - mean(bootcoefs) )>= abs(bootcoefs))
}
pvals <- apply(L0glm_infer_out$boot.result$t,2,twosidep) # 2-sided p values
sum(pvals<0.05)
library(car)
stats_L0glm = data.frame(cbind(L0glm_infer_out$boot.result$t0,confint(L0glm_infer_out$boot.result,type="perc"),pvals,L0glm_infer_out$p.value))
colnames(stats_L0glm)=c("coef","lower","upper","p.value","p.value.chris")
stats_L0glm
# check # from https://stats.stackexchange.com/questions/20701/computing-p-value-using-bootstrap-with-r
# https://stats.stackexchange.com/questions/83012/how-to-obtain-p-values-of-coefficients-from-bootstrap-regression
library(ggplot2)
library(tidyr)
df = gather(data.frame(L0glm_infer_out$boot.result$t))
df$key = factor(df$key, levels=paste0("X",1:ncol(L0glm_infer_out$boot.result$t)), label=rownames(stats_L0glm))
ggplot(df, aes(value)) +
  geom_histogram(bins = 10) +
  facet_wrap(~key, scales = 'free_x')
# or use psych::multi.hist ?


##### Comparison with nnls, l0ara, L0Learn, nnLasso (glmnet) solutions ####
library(l0ara) # used my fork with nonnegativity constraints here
# library(devtools)
# install_github("hazimehh/L0Learn", ref="nng") # L0Learn with nonnegativity constraints, https://github.com/hazimehh/L0Learn/issues/43
library(L0Learn)
library(nnls)
library(glmnet)
library(microbenchmark)
ctrl.fit <- control.fit.gen(tol=1E-3) # default
ctrl.iwls <- control.iwls.gen(maxit = 1, rel.tol=1E-3)
ctrl.l0 <- control.l0.gen(rel.tol=1E-3) # default
microbenchmark("lmfit" = {beta_lmfit <- lm.fit(x = sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)))$coefficients},
               "nnls" = {beta_nnls <- nnls(A = sim$X*sqrt(1/(sim$y+1)), b=sim$y*sqrt(1/(sim$y+1)))$x},
               "nnl0ara" = {beta_l0ara <- l0ara(x = sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                   family="gaussian", lam=1, standardize=FALSE, maxit=1E4, eps=1E-4)$beta},
               "nnL0glm" = {L0glm_fit <- L0glm.fit(X = sim$X, y = sim$y,
                                        family = poisson(identity),
                                        lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                        control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                                        control.fit = ctrl.fit)
                 beta_L0glm <- L0glm_fit$coefficients},
               "nnL0Learncv" = {L0Learn_cvfit <- L0Learn.cvfit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                               nFolds = 3,
                                               loss = "SquaredError",
                                               penalty="L0", algorithm="CD", nLambda=50,
                                               maxSuppSize=50, intercept=FALSE, autoLambda=TRUE ) },
               "nnL0Learnfit" = {L0Learn_fit <- L0Learn.fit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                           loss = "SquaredError",
                                           penalty="L0", algorithm="CD",
                                           maxSuppSize=50, intercept=FALSE, nLambda=1,
                                           autoLambda=FALSE, lambdaGrid=list(1/(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2)) ) }, # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
               "nnLASSO" = {lam.max <- function (X, y) max( abs(crossprod(X,y)) ) / nrow(X) # largest lambda value for LASSO so that no variables would be selected, cf https://stats.stackexchange.com/questions/166630/glmnet-compute-maximal-lambda-value & https://stats.stackexchange.com/questions/292147/how-to-find-the-smallest-lambda-such-that-all-lasso-elastic-net-coefficient?noredirect=1&lq=1
               lammax <- lam.max(X=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)))
               nnLASSO_cvfit <- cv.glmnet(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)), family="gaussian",
                                           lambda = exp(seq(log(lammax),-20,length.out=50)),
                                            intercept=F, standardize=F, nfolds=3, lower.limits=0) # 26 ms
               # plot(nnLASSO_cvfit)
               # matplot(x=as.matrix(nnLASSO_cvfit$glmnet.fit$lambda,ncol=1), y=t(nnLASSO_cvfit$glmnet.fit$beta), type="l", log="x")
               beta_nnLASSO <- coef(nnLASSO_cvfit, s = nnLASSO_cvfit$lambda.1se)[-1] # nnLASSO_cvfit$lambda.1se / nnLASSO_cvfit$lambda.min
               },
               times=5)
# nnL0Learn coefficient path
matplot(x=as.matrix(L0Learn_cvfit$fit$lambda[[1]],ncol=1), y=t(L0Learn_cvfit$fit$beta[[1]]),type="l",log="x",xlab="Lambda",ylab="Coefficients")

plot(sim$beta_true[sim$beta_true!=0], L0glm_fit$coefficients[sim$beta_true!=0], pch = 16, col="steelblue",
     main="L0GLM fitted vs expected coefficients", xlab="Expected true coefficients", ylab="Fitted coefficients", log="xy")
norm(sim$beta_true-L0glm_fit$coefficients,"2")
sum(L0glm_fit$coefficients>0) # 13
plot(sim$beta_true[sim$beta_true!=0], beta_l0ara[sim$beta_true!=0], pch = 16, col="steelblue",
     main="l0ara fitted vs expected coefficients", xlab="Expected true coefficients", ylab="Fitted coefficients", log="xy")
norm(sim$beta_true-beta_l0ara,"2") # slightly better solution than L0glm & faster...
min(beta_l0ara) # 0
sum(beta_l0ara>0) # 12
beta_L0Learn = coef(L0Learn_fit)
plot(sim$beta_true[sim$beta_true!=0], beta_L0Learn[sim$beta_true!=0], pch = 16, col="steelblue",
     main="L0Learn fitted vs expected coefficients", xlab="Expected true coefficients", ylab="Fitted coefficients", log="xy")
norm(sim$beta_true-beta_L0Learn,"2")
sum(beta_L0Learn>0) # 18
optlambda_L0Learn = L0Learn_cvfit$fit$lambda[[1]][which.min(L0Learn_cvfit$cvMeans[[1]])]
optlambda_L0Learn # 2.9E-6

# note:cvMeans of L0Learn doesn't properly take into account our weights here, so let's do this properly:
coefs_L0Learn = do.call(cbind,lapply(L0Learn_cvfit$fit$lambda[[1]], function (lam) coef(L0Learn_cvfit, lambda=lam, gamma=0) ))
weighted.rmse <- function(actual, predicted, weight){
  sqrt(sum((predicted-actual)^2*weight)/sum(weight))
}
weightedrmse_L0Learn = apply(coefs_L0Learn, 2, function (fitted_coefs) {
  pred = (sim$X*sqrt(1/(sim$y+1))) %*% fitted_coefs
  weighted.rmse(actual=sim$y_true,
              predicted=pred,
              weight=1/(sim$y+1)) } )

plot(L0Learn_cvfit$fit$lambda[[1]], weightedrmse_L0Learn, type="l", log="x")
optlambda_L0Learn_weightedrmse = L0Learn_cvfit$fit$lambda[[1]][which.min(weightedrmse_L0Learn)]
optlambda_L0Learn_weightedrmse # 8E-6
abline(v = optlambda_L0Learn_weightedrmse, col="red")
1/(norm((sim$X*sqrt(1/(sim$y+1)))[,floor(ncol(sim$X)/2)],"2")*norm(sim$y*sqrt(1/(sim$y+1)),"2"))
(norm((sim$X*sqrt(1/(sim$y+1)))[,floor(ncol(sim$X)/2)],"2")/norm(sim$y*sqrt(1/(sim$y+1)),"2"))^2 # 1E-7
1/norm(sim$y*sqrt(1/(sim$y+1)),"2")
Xw=sim$X*sqrt(1/(sim$y+1))
yw=sim$y*sqrt(1/(sim$y+1))
Xn=apply(Xw, 2, function (col) col/norm(col,"2"))
yn=yw/norm(yw,"2")
1/sum(t(Xw)%*%Xw)^2 # 1E-7
norm(sim$y*sqrt(1/(sim$y+1)),"2") / sum(t(Xw)%*%Xw)^2

mean(as.vector((crossprod(Xn,yn))^2) / as.vector(diag(crossprod(Xn,Xn)))) /
(mean(as.vector((crossprod(sim$X*sqrt(1/(sim$y+1)),sim$y*sqrt(1/(sim$y+1))))^2) /
       as.vector(diag(crossprod(sim$X*sqrt(1/(sim$y+1)),sim$X*sqrt(1/(sim$y+1)))))))
# 9.622673e-06 - correct lambda scaling factor in L0Learn compared to in L0glm, see https://stats.stackexchange.com/questions/416144/minimum-and-maximum-regularization-in-l0-pseudonorm-penalized-regression
(as.vector((crossprod(Xn,yn))^2) / as.vector(diag(crossprod(Xn,Xn)))) /
  ((as.vector((crossprod(sim$X*sqrt(1/(sim$y+1)),sim$y*sqrt(1/(sim$y+1))))^2) /
          as.vector(diag(crossprod(sim$X*sqrt(1/(sim$y+1)),sim$X*sqrt(1/(sim$y+1)))))))
1/(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2) # 9.622673e-06


norm((sim$X*sqrt(1/(sim$y+1)))[,floor(ncol(sim$X)/2)],"2")^2
norm(sim$y*sqrt(1/(sim$y+1)),"2")

beta_L0Learn_cvfit = coef(L0Learn_cvfit, lambda=optlambda_L0Learn_weightedrmse, gamma=0)
norm(sim$beta_true-beta_L0Learn_cvfit,"2")
sum(beta_L0Learn_cvfit>0) # 12 - so 2 false positives
plot(sim$beta_true[sim$beta_true!=0], beta_L0Learn_cvfit[sim$beta_true!=0], pch = 16, col="steelblue",
     main="cv L0Learn fitted vs expected coefficients", xlab="Expected true coefficients", ylab="Fitted coefficients", log="xy")
plot(sim$beta_true[sim$beta_true!=0], beta_nnLASSO[sim$beta_true!=0], pch = 16, col="steelblue",
     main="cv nonnegative LASSO (glmnet) fitted vs expected coefficients", xlab="Expected true coefficients", ylab="Fitted coefficients", log="xy")
sum(beta_nnLASSO>0) # 28


plot_benchmark(x=sim$x, y=sim$y, fit=L0glm_fit, beta_true = sim$beta_true, main="Observed signal (black) and true (red) and\ninferred spike train (blue=L0glm, green=nnl0ara, orange=nnL0Learn)")
lines(x=seq(min(sim$x),max(sim$x),length.out=p)+0.5, y=-beta_l0ara, type="h", col="green3", lwd=2) # nnl0ara estimates
lines(x=seq(min(sim$x),max(sim$x),length.out=p)+1, y=-beta_L0Learn_cvfit, type="h", col="orange", lwd=2) # L0Learn estimates


# TO DO
# check if coefficients are scaled back correctly when normalize = TRUE
# change p values inference to prop that are zero ??
# change lm.fit to solve() & remove all parts based on row augmentation?




## BENCHMARK SINGLE ITERATION OF ADAPTIVE RIDGE ALGO ####

sd_y <- sqrt(var(Y)*(n-1)/n)

library(microbenchmark)

lmridge_solve = function (X, Y, lambda) solve(crossprod(X)+lambda*diag(ncol(X)), crossprod(X,Y))[,1] # cf Liu & Li 2016 algo 1, see Liu et al 2017 & https://github.com/tomwenseleers/l0ara/blob/master/src/l0araC.cpp for application to GLMs
lmridge_solve_largep = function (X, Y, lambda) (t(X) %*% solve(tcrossprod(X)+lambda*diag(nrow(X)), Y))[,1] # cf Liu & Li 2016 algo 2, see Liu et al 2017 & https://github.com/tomwenseleers/l0ara/blob/master/src/l0araC.cpp for application to GLMs
lmridge_qrsolve = function (X, Y, lambda) qr.solve(crossprod(X)+lambda*diag(ncol(X)), crossprod(X,Y))[,1] # solve using QR decomposition
lmridge_qrsolve_largep = function (X, Y, lambda) (t(X) %*% qr.solve(tcrossprod(X)+lambda*diag(nrow(X)), Y))[,1]
chol.solve = function (a, b) { ch <- chol(crossprod(a)) # solve using Cholesky decomposition
backsolve(ch, forwardsolve(ch, crossprod(a, b), upper = TRUE, trans = TRUE)) }
lmridge_cholsolve = function (X, Y, lambda) chol.solve(crossprod(X)+lambda*diag(ncol(X)), crossprod(X,Y))
lmridge_cholsolve_largep = function (X, Y, lambda) (t(X) %*% chol.solve(tcrossprod(X)+lambda*diag(nrow(X)), Y))
lmridge_lmfit = function (X, Y, lambda) coef(lm.fit(crossprod(X)+lambda*diag(ncol(X)), crossprod(X,Y)))
lmridge_lmfit_largep = function (X, Y, lambda) { coefs = coef(lm.fit(tcrossprod(X)+lambda*diag(nrow(X)), Y))
                                                 coefs[is.na(coefs)] = 0
                                                 t(X) %*% coefs }
library(nnls)
lmridge_nnls = function (X, Y, lambda) nnls(A=crossprod(X)+lambda*diag(ncol(X)), b=crossprod(X,Y))$x # works for n>p and p>n
lmridge_nnls_largep = function (X, Y, lambda) t(X) %*% nnls(A=tcrossprod(X)+lambda*diag(nrow(X)), b=Y)$x # didn't work for me...
lmridge_nnls_rbind = function (X, Y, lambda) nnls(A=rbind(X,sqrt(lambda)*diag(ncol(X))), b=c(Y,rep(0,ncol(X))))$x # other nnls solution above is more accurate, see https://stats.stackexchange.com/questions/203685/how-to-perform-non-negative-ridge-regression
lmridge_glmnet = function (X, Y, lambda, nonnegative = FALSE) { # -> BEST OPTION FOR n=p
  require(glmnet)
  n = nrow(X)
  sd_Y <- sqrt(var(Y)*(n-1)/n)
  if (nonnegative) { lower.limits = 0 } else { lower.limits = -Inf }
  fit = glmnet(X, Y, alpha=0, standardize = FALSE, intercept = FALSE, lambda = sd_Y*lambda/n,
               thresh = 1e-12, lower.limits = lower.limits, maxit = 1E5)  #
  as.vector(coef(fit, s = sd_Y*lambda/n, exact = TRUE, intercept = FALSE, x=X, y=Y, thresh = 1e-12, lower.limits=lower.limits))[-1] #
}
lmridge_glmnet2 = function (X, Y, lambda, nonnegative = FALSE) {
  require(glmnet)
  x = crossprod(X)+lambda*diag(ncol(X))
  y = crossprod(X,Y)
  if (nonnegative) { lower.limits = 0 } else { lower.limits = -Inf }
  fit = glmnet(x, y, alpha=1, standardize = FALSE, intercept = TRUE, lambda = 0,
               thresh = 1e-12, lower.limits = lower.limits, maxit = 1E5)
  as.vector(coef(fit, s = 0, exact = TRUE, intercept = TRUE, alpha=1, x=x, y=y, thresh = 1e-12, lower.limits=lower.limits))[-1] #
}
lmridge_glmnet3 = function (X, Y, lambda, nonnegative = FALSE) {
  require(glmnet)
  x = tcrossprod(X)+lambda*diag(nrow(X))
  if (nonnegative) { lower.limits = 0 } else { lower.limits = -Inf }
  fit = glmnet(x=x, y=Y, alpha=1, standardize = FALSE, intercept = FALSE, # lambda = 0,
               thresh = 1e-12, lower.limits = lower.limits, maxit = 1E5)
  as.vector(t(X) %*% coef(fit, s = 0, exact = TRUE, intercept = FALSE, alpha=1, standardize = FALSE, x=x, y=Y, thresh = 1e-12, lower.limits=lower.limits)[-1]) #
}
library(NNLM)
lmridge_nnlm = function (X, Y, lambda, nonnegative = FALSE) {
  require(NNLM)
  x = crossprod(X)+lambda*diag(ncol(X))
  y = crossprod(X,Y)
  fit = nnlm(x, y, alpha=c(0,0,0), method="scd", loss="mse",
             init=NULL, check.x=FALSE, rel.tol = 1e-20)
  as.vector(coef(fit)) #
}
lmridge_nnlm_largep = function (X, Y, lambda, nonnegative = FALSE) {
  require(NNLM)
  x = tcrossprod(X)+lambda*diag(nrow(X))
  fit = nnlm(x, Y, alpha=c(0,0,0), method="scd", loss="mse",
             init=NULL, check.x=FALSE, rel.tol = 1e-20)
  as.vector(t(X) %*% coef(fit)) #
}

library(microbenchmark)
n <- 500
p <- 1000
s <- 0.1 # sparsity as proportion of p that have nonzero coefficients
k <- round(p*s) # nr of nonzero covariates
sim <- simulate_spike_train(n = n, p = p, k = k,
                            mean_beta = 1000, sd_logbeta = 1,
                            family = "poisson", seed = 123, Plot = TRUE)
X = sim$X
Y = sim$y
beta = sim$beta_true
lambda <- lambda_ridge <- 1E-10


microbenchmark(
  "lmridge_solve" = {
    beta_lmridge_solve <- lmridge_solve(X, Y, lambda_ridge)
  },
  # "lmridge_solve_largep" = {
  #   beta_lmridge_solve_largep <- lmridge_solve_largep(X, Y, lambda_ridge)
  # },
  "lmridge_qrsolve" = {
    beta_lmridge_qrsolve <- lmridge_qrsolve(X, Y, lambda_ridge)
  },
  # "lmridge_qrsolve_largep" = {
  #   beta_lmridge_qrsolve_largep <- lmridge_qrsolve_largep(X, Y, lambda_ridge)
  # },
  "lmridge_cholsolve" = {
    beta_lmridge_cholsolve <- lmridge_cholsolve(X, Y, lambda_ridge)
  },
  # "lmridge_cholsolve_largep" = {
  #   beta_lmridge_cholsolve_largep <- lmridge_cholsolve_largep(X, Y, lambda_ridge)
  # },
  "lmridge_lmfit" = {
    beta_lmridge_lmfit <- lmridge_lmfit(X, Y, lambda_ridge)
  },
  # "lmridge_lmfit_largep" = {
  #   beta_lmridge_lmfit_largep <- lmridge_lmfit_largep(X, Y, lambda_ridge)
  # },
  "lmridge_nnls" = {
    beta_lmridge_nnls <- lmridge_nnls(X, Y, lambda_ridge)
  },
  # "lmridge_nnls_largep" = {
  #   beta_lmridge_nnls_largep <- lmridge_nnls_largep(X, Y, lambda_ridge)
  # },
  "lmridge_nnls_rbind" = {
    beta_lmridge_nnls_rbind <- lmridge_nnls_rbind(X, Y, lambda_ridge)
  },
  "lmridge_glmnet" = {
    beta_lmridge_glmnet <- lmridge_glmnet(X, Y, lambda_ridge, nonnegative=FALSE)
  },
  "lmridge_glmnet2" = {
    beta_lmridge_glmnet2 <- lmridge_glmnet2(X, Y, lambda_ridge, nonnegative=FALSE)
  },
  "lmridge_glmnet3" = {
    beta_lmridge_glmnet3 <- lmridge_glmnet3(X, Y, lambda_ridge, nonnegative=FALSE)
  },
  "lmridge_nnglmnet" = {
    beta_lmridge_nnglmnet <- lmridge_glmnet(X, Y, lambda_ridge, nonnegative=TRUE)
  },
  "lmridge_nnglmnet2" = {
    beta_lmridge_nnglmnet2 <- lmridge_glmnet2(X, Y, lambda_ridge, nonnegative=TRUE)
  },
  "lmridge_nnglmnet3" = {
    beta_lmridge_nnglmnet3 <- lmridge_glmnet3(X, Y, lambda_ridge, nonnegative=TRUE)
  },
  "lmridge_nnlm" = {
    beta_lmridge_nnlm <- lmridge_nnlm(X, Y, lambda_ridge)
  },
  # "lmridge_nnlm_largep" = {
  #   beta_lmridge_nnlm_largep <- lmridge_nnlm_largep(X, Y, lambda_ridge)
  # },
  "L0glm (ridge settings)" = {
    beta_L0glm_fit <- coef(L0glm.fit(X = X, y = Y,
                                     family = gaussian(),
                                     lambda = lambda_ridge, nonnegative = FALSE, normalize = FALSE,
                                     control.iwls = list(maxit = 1, rel.tol
                                                         = 1e-04, thresh = 1e-05, warn = FALSE),
                                     control.l0 = list(maxit = 1, rel.tol = 1e-04, delta =
                                                         1e-05, gamma = 2, warn = FALSE),
                                     control.fit = list(maxit = 1,
                                                        block.size = NULL, tol = 1e-07)))
  },
  "nnL0glm (nonnegative ridge settings)" = {
    beta_nnL0glm_fit <- coef(L0glm.fit(X = X, y = Y,
                                       family = gaussian(),
                                       lambda = lambda_ridge, nonnegative = TRUE, normalize = FALSE,
                                       control.iwls = list(maxit = 1, rel.tol
                                                           = 1e-04, thresh = 1e-05, warn = FALSE),
                                       control.l0 = list(maxit = 1, rel.tol = 1e-04, delta =
                                                           1e-05, gamma = 2, warn = FALSE),
                                       control.fit = list(maxit = 1,
                                                          block.size = NULL, tol = 1e-07)))
  },
  times = 1
)

cbind(beta,beta_lmridge_solve, #beta_lmridge_solve_largep,
      beta_lmridge_qrsolve, # beta_lmridge_qrsolve_largep,
      beta_lmridge_cholsolve, # beta_lmridge_cholsolve_largep,
      beta_lmridge_lmfit, # beta_lmridge_lmfit_largep,
      beta_lmridge_nnls, # beta_lmridge_nnls_largep,
      beta_lmridge_nnls_rbind,
      beta_lmridge_glmnet, beta_lmridge_glmnet2, beta_lmridge_glmnet3,
      beta_lmridge_nnglmnet, beta_lmridge_nnglmnet2, beta_lmridge_nnglmnet3,
      beta_lmridge_nnlm, # beta_lmridge_nnlm_largep,
      beta_L0glm_fit, beta_nnL0glm_fit)[1:10,]
# unconstrained solutions
c(norm(beta-beta_lmridge_solve,"2")) # best & fastest option for n >= p
c(norm(beta-beta_lmridge_solve_largep,"2")) # best & fastest option for p > n
c(norm(beta-beta_lmridge_qrsolve,"2"))
c(norm(beta-beta_lmridge_qrsolve_largep,"2"))
c(norm(beta-beta_lmridge_cholsolve,"2"))
c(norm(beta-beta_lmridge_cholsolve_largep,"2"))
c(norm(beta-beta_lmridge_lmfit,"2"))
c(norm(beta-beta_lmridge_lmfit_largep,"2"))
c(norm(beta-beta_lmridge_glmnet,"2")) # best unconstr glmnet option for n>=p, but very slow
c(norm(beta-beta_lmridge_glmnet2,"2"))
c(norm(beta-beta_lmridge_glmnet3,"2")) # best unconstr glmnet option for p > n, but very slow
c(norm(beta-beta_L0glm_fit,"2"))
# nonnegativity constrained
c(norm(beta-beta_lmridge_nnls,"2")) # -> best nnls option for p > n & n >= p
c(norm(beta-beta_lmridge_nnls_largep,"2")) # doesn't work for p > n or n >= p
c(norm(beta-beta_lmridge_nnls_rbind,"2"))
c(norm(beta-beta_lmridge_nnglmnet,"2")) # -> best nnglmnet nonnegative option for n>=p but very slow
c(norm(beta-beta_lmridge_nnglmnet2,"2")) # -> best nnglmnet nonnegative option for p > n & fastest
c(norm(beta-beta_lmridge_nnglmnet3,"2"))
c(norm(beta-beta_lmridge_nnlm,"2"))
c(norm(beta-beta_lmridge_nnlm_largep,"2"))
c(norm(beta-beta_nnL0glm_fit,"2"))




####---- BENCHMARK WITH GLM ----####


# From ?glm examples
# Dobson (1990) Page 93: Randomized Controlled Trial :
counts <- c(18,17,15,20,10,20,25,13,12)
outcome <- gl(3,1,9)
treatment <- gl(3,3)

# Becnhmark
microbenchmark(
  # Glm fitting
  "glm" = {
    glm_fit <- glm(counts ~ outcome + treatment, family = poisson())
  },
  # L0glm fitting (using glm settings)
  "L0glm (glm settings)" = {
    L0glm_fit <- L0glm(counts ~ outcome + treatment,
                       family = poisson(),
                       lambda = 0, tune.meth = "none", nonnegative = FALSE,
                       control.iwls = list(maxit = 25, thresh = .Machine$double.eps),
                       control.l0 = list(maxit = 1),
                       control.fit = list(maxit = 1), verbose = FALSE)
  },
  times = 25
)

# Compare coefficients
df <- data.frame(coef.glm = coef(glm_fit),
                 coef.L0glm = coef(L0glm_fit))
abs(df$coef.glm - df$coef.L0glm)

# Plot coefficients
data <- data.frame(y = unlist(df),
                   x = rep(1:nrow(df), ncol(df)),
                   type = rep(c("glm", "L0glm"), each = nrow(df)))
pl <- ggplot(data = data, aes(x = x, y = y, color = type)) +
  geom_point() + geom_line() +
  ggtitle("Compare coefficients estimated using glm or L0glm") +
  ylab("Estimate") + xlab("Index") +
  scale_colour_manual(name = "Algorithm",
                      values = c(glm = "red3", L0glm = "green4"),
                      labels = c(glm = "glm", L0glm = "L0glm"))
graph2ppt(pl, file = "Github/graphs", scaling = 50, append = TRUE)

# Conclusion
# Both algorithms give almost exactly the same solution (up to 2E-15). The
# higher functionnality of the L0glm framework (possibility of nonnegative
# constraints and regularization) is a the cost of timing performance.


####---- BENCHMARK RIDGE REGRESSION WITH GLMNET, RIDGE AND PENALIZED ----####

library(glmnet)
library(ridge)
library(penalized)

# Simulate data with Gaussian noise
set.seed(123)
n <- 100
p <- 20
x <- matrix(rnorm(n*p), nrow = n, ncol = p)
x <- scale(x)
beta <- runif(p, min = -1)
y0 <- x %*% beta
y <- y0 + rnorm(n, mean = 0, sd = 2.5)
y <- scale(y)
lam <- 1

microbenchmark(
  # Ridge regression using glmnet
  "glmnet" = {
    glmnet_fit <- glmnet(x = x, y = y, family = "gaussian", alpha = 0,
                         standardize = FALSE, thresh = .Machine$double.eps,
                         lambda = 10^seq(10,log10(lam)), intercept = TRUE)
  },
  # Ridge regression using ridge
  "ridge" = {
    ridge_fit <- linearRidge(y ~ 1 + ., data = data.frame(y = y, x),
                             lambda = lam*n, scaling = "none")
  },
  "penalized" = {
    penal_fit <- penalized(response = y, penalized = x,
                           lambda1 = 0, lambda2 = lam*n, positive = FALSE, model= "linear",
                           epsilon = .Machine$double.eps, maxiter = 25, trace = F)
  },
  # L0glm fitting (using glm settings)
  "L0glm (ridge settings)" = {
    L0glm_fit <- L0glm(y ~ 1 + ., data = data.frame(y = y, x),
                       family = gaussian(),
                       lambda = lam*n, tune.meth = "none", nonnegative = FALSE, normalize = FALSE,
                       control.iwls = list(maxit = 25, thresh = .Machine$double.eps),
                       control.l0 = list(maxit = 1),
                       control.fit = list(maxit = 1, tol = .Machine$double.eps),
                       verbose = FALSE)
  },
  times = 25
)
# Note: all lambdas except for glmnet are scaled with 'n'. This is to allow
# comparing the same objective function as illustrated in https://stackoverflow.com/questions/39863367/ridge-regression-with-glmnet-gives-different-coefficients-than-what-i-compute

x_norm = sweep(x, 2, sapply(1:ncol(x), function (col) norm(x[,col],"2")), FUN="/")
y_norm = y/norm(y,"2")
lambda_ridge = 1
microbenchmark(fit <- L0Learn.fit(x = x_norm, y = y_norm, loss = "SquaredError", penalty = "L0L2",
              algorithm = "CDPSI", maxSuppSize = n, nLambda = 1, nGamma = 1,
              gammaMin = lambda_ridge, gammaMax = lambda_ridge,
              activeSet = TRUE,
              autoLambda = FALSE,
              lambdaGrid = as.list(0),
              intercept = FALSE), time=100)
coef(fit)
ridge_fit <- coef(linearRidge(y_norm ~ 0 + ., data = data.frame(y_norm = y_norm, x_norm),
                         lambda = lambda_ridge*n, scaling = "none"))
coef(ridge_fit)



set.seed(123)
n    <- 1000
p   <-  1000
lambda_ridge <- lambda <- 10
X <- matrix(rnorm(n*p,0,10),n,p)
X[X<0] <- 0
beta <- 10^rnorm(p,0,1)
Y    <- X%*%beta+rnorm(n,0,0.1)
Y[Y<0] <- 0



# from l0ara:
# to implement:
# Xt = x;
# xw = x*w;
# z = A*xw + y-s1;
# if (n > p) {
#   w = solve(trans(Xt)*A*x+lam*eye(p,p), trans(Xt)*z);
# } else {
#   w = trans(Xt)*solve(A*x*trans(Xt)+lam*eye(n,n), z);
# }
# Xt = repmat(trans(w % w), n, 1) % x;



library(glmnet)
microbenchmark(fit_glmnet <- glmnet(X, Y, alpha=1E-10, standardize = F, intercept = FALSE,
                                    thresh = 1e-7, lambda = sd_y*lambda_ridge/n), times=1) # 81ms / 4.6s with thresh=1E-7
beta_glmnet <- as.vector(coef(fit_glmnet, s = sd_y*lambda_ridge/n, exact = TRUE, x=X, y=Y))[-1]
library(ridge)
microbenchmark(beta_ridge <- coef(linearRidge(y ~ 0 + ., data = data.frame(y = Y, X),
                                              lambda = lambda_ridge, scaling = "none")), times=1) # 881 ms
cbind(beta_theor1[1:10], beta_theor2[1:10], beta_theor3[1:10], beta_theor4[1:10], beta_theor5[1:10], beta_theor6[1:10],
      beta_glmnet[1:10], beta_ridge[1:10])

library(speedglm)
system.time(beta_theor7<-speedlm.fit(X=X_aug, y=Y_aug, sparse=FALSE,
                                     eigendec=FALSE, method="eigen", model=FALSE, fitted=FALSE)$coefficients) # 39s with n=p=10000 - FASTEST! BUT ONLY WITH INTEL MKL
system.time(beta_theor7b<-speedlm.fit(X=X_aug, y=Y_aug, sparse=FALSE, eigendec=FALSE, method="Cholesky", model=FALSE, fitted=FALSE)$coefficients) # 39s with n=p=10000 BUT ONLY WITH INTEL MKL
system.time(beta_theor8<-speedlm.fit(X=X_aug_sparse, y=Y_aug, sparse=TRUE, eigendec=FALSE, method="Cholesky", model=FALSE, fitted=FALSE)$coefficients) # very slow, so sparsity doesn't help...






library(glmnet)
lambda_ridge = 1
sd_y <- sqrt(var(y)*(n-1)/n)[1,1]
fit_glmnet_ridge <- glmnet(x, y, standardize = F, alpha = 0, intercept = FALSE, family="gaussian", thresh = 1e-20)
beta_glmnet <- as.vector(coef(fit_glmnet_ridge, s = 2*sd_y*10/n, x=x, y=y, exact = TRUE))[-1]
coef(linearRidge(y ~ 0 + ., data = data.frame(y = y, x),
                 lambda = lambda_ridge*n, scaling = "none"))

ridge.fit.lambda <- ridge.fit.cv$lambda.1se
ridge.coef <- (coef(ridge.fit.cv, s = ridge.fit.lambda))[-1]
ridge.coef.DEF <- drop(solve(crossprod(x_norm) + diag(n * ridge.fit.lambda, p), crossprod(x_norm, y_norm)))


# Check results
df <- data.frame(coef.glmnet = as.vector(coef(glmnet_fit, s = 1)),
                 coef.ridge = coef(ridge_fit),
                 coef.penalized = coef(penal_fit),
                 coef.L0glm = coef(L0glm_fit),
                 coef.true = c(0, beta)) # no intercept is present
abs(df$coef.glmnet - df$coef.L0glm)
abs(df$coef.ridge - df$coef.L0glm)
abs(df$coef.penalized - df$coef.L0glm)

# Plot coefficients
data <- data.frame(y = unlist(df),
                   x = rep(1:nrow(df), ncol(df)),
                   type = rep(c("glmnet", "ridge", "penalized", "L0glm", "true"), each = nrow(df)))
pl <- ggplot(data = data, aes(x = x, y = y, color = type)) +
  geom_point() + geom_line() +
  ggtitle("Compare true coefficients with coefficients estimated \nusing glmnet, ridge, or L0glm") +
  ylab("Estimate") + xlab("Index") +
  scale_colour_manual(name = "Algorithm",
                      values = c(glmnet = "red3", ridge = "orange2", penalized = "purple", L0glm = "green4", true = "grey40"),
                      labels = c(glmnet = "glmnet", ridge = "ridge", penalized = "penalized", L0glm = "L0glm", true = "True"))
graph2ppt(pl, file = "Github/graphs", scaling = 50, append = TRUE)

# Conclusion
# Both algorithms exhibit coefficients following the same trend but the absolute
# coefficients values are quite different. However, L0glm seems to better
# estimate the true value of the coefficients

# TODO delete, optimizing lambda
test1 <- cv.glmnet(x = x, y = y, family = "gaussian", alpha = 0,
                   standardize = FALSE, thresh = .Machine$double.eps,
                   nfolds = 3,
                   lambda = 10^seq(10,-10), intercept = FALSE)
plot(test1$lambda, test1$cvm,log= "xy", type = 'l')
test1$lambda.min
test2 <- L0glm(y ~ 0 + ., data = data.frame(y = y, x), family = gaussian(),
               lambda = 10^seq(10,-10), tune.meth = "3-fold", tune.crit = "rss",
               nonnegative = FALSE, control.l0 = list(maxit = 1),
               control.iwls = list(maxit = 25, thresh = .Machine$double.eps),
               control.fit = list(maxit = 1), verbose = TRUE)
plot(test2$lambda.tune$lambdas, test2$lambda.tune$IC[, "rss"],log= "xy", type = 'l')
test2$lambda.tune$best.lam


####---- BENCHMARK WITH L0Learn, L0ara, bestsubset ----####

library(L0Learn)
library(l0ara)
library(bestsubset)

# Modified from ?L0Learn.fit examples
# Generate synthetic data for this example
GenSynthetic = function (n, p, k, seed = 123, mean_beta = 1, sd_beta = 0, sd_noise = 1, sd_X = 0.1) {
  set.seed(seed)
  X = matrix(rnorm(n * p, mean = 0, sd  = sd_X), nrow = n, ncol = p)
  intercept = FALSE # we have no intercept in this problem
  beta_true = rep(0, p)
  beta_true[1:k] = rnorm(k, mean = mean_beta, sd = sd_beta)
  e = rnorm(n, mean = 0, sd = sd_noise)
  # e_validation = rnorm(n, mean = 0, sd = sd_noise)
  y_true = X %*% beta_true
  y = y_true + e
  # y_validation = y_true + e_validation

  if (intercept) { meany = mean(y)
                   meancolsX = sapply(1:ncol(X), function (i) mean(X[,i])) } else {
                     meany = 0
                     meancolsX = rep(0,ncol(X))
                   }
  X_norm = sweep(X, 2, meancolsX, "-")
  X_scaling = sapply(1:ncol(X), function (i) norm(X_norm[,i],"2"))
  X_norm = sweep(X, 2, X_scaling, "/")
  # scaling factor required to scale coefficients back to original scale before centering & L2 normalization:
  y_scaling = (norm(y-meany,"2"))
  coef_scaling = y_scaling/X_scaling
  y_norm = (y-meany)/(norm(y-meany,"2"))

  list(X = X, y = y, X_norm = X_norm, y_norm = y_norm,
       y_true = y_true, beta_true = beta_true,
       y_scaling = y_scaling, coef_scaling = coef_scaling)
}

n <- 200
p <- 500
k <- 10
mean_beta <- 1
sd_beta <- 0
sd_noise <- 1
sd_X <- 1
data <- GenSynthetic(n = n, p = p, k = k, seed = 123, mean_beta = mean_beta, sd_beta = sd_beta,
                     sd_noise = sd_noise, sd_X = sd_X)
x <- data$X
beta_true <-  data$beta_true
y <- data$y
x_norm <- data$X_norm
y_norm <- data$y_norm
y_scaling <- data$y_scaling
coef_scaling <- data$coef_scaling
# scaling factor required to scale coefficients back to original scale before centering & L2 normalization

# Tests to Select lambda & plot coefficients

# best lambda of L0Learn = true lambda in L0 penalized LS regression (1/2)*RSS + lambda*L0norm
# best lambda of l0ara = lambda of L0Learn * factor of (1/4) of Frommlet
# best lambda of L0glm = lambda of L0Learn * factor of 2 (present in objective of l0ara) * factor of (1/4) of Frommlet

# L0Learn
microbenchmark(test1 <- L0Learn.cvfit(x = x_norm, y = y_norm, loss = "SquaredError", penalty = "L0",
                                      algorithm = "CDPSI", maxSuppSize = 40, nLambda = 100, nGamma = 0,
                                      activeSet = TRUE,
                                      maxIters = 200, tol = 1e-7, autoLambda = TRUE,
                                      # lambdaGrid = as.list(rev(lambdas)),
                                      intercept = FALSE,
                                      nFolds = 10, seed = 123
                                    ), times=1) # 117 ms
plot(test1)
plot(test1$fit$lambda[[1]], test1$cvMeans[[1]], log = "xy", type = "l")
abline(v=test1$fit$lambda[[1]][which.min(test1$cvMeans[[1]])], col="red")
test1$fit$lambda[[1]][which.min(test1$cvMeans[[1]])]
# best lambda = 0.00156544 if X & y are first centered & L2 norm normalized
# = true lambda that maximizes L0 penalized regression
min(test1$cvMeans[[1]]) # 0.0002415368 = minimum MSE
plot(beta_true,type="l")
lines(coef_scaling*coef(test1, lambda=test1$fit$lambda[[1]][which.min(test1$cvMeans[[1]])], gamma=0), col="red", type="l")
# support size in function of lambda
plot(test1$fit$lambda[[1]], test1$fit$suppSize[[1]], log = "x", type = "l")

# coefficient path:
test1_lambda.coefs = coef_scaling*t(do.call(cbind,lapply(test1$fit$lambda[[1]], function (lam) coef(test1, lambda=lam))))
matplot(test1$fit$lambda[[1]], test1_lambda.coefs, type="l", log="x")
abline(v=test1$fit$lambda[[1]][which.min(test1$cvMeans[[1]])], col="red")

# max lambda that causes all variables to be selected out of the model with or without intercept, see
# https://stats.stackexchange.com/questions/416144/minimum-and-maximum-regularization-in-l0-pseudonorm-penalized-regression
max(test1$fit$lambda[[1]])
maxlambda = max((crossprod(x_norm, y_norm)^2)/2)
maxlambda # = 0.06448368




lambdas <- 10^seq(log10(min(test1$fit$lambda[[1]])),log10(max(test1$fit$lambda[[1]])), length.out = 50)

# lambdas <- 10^seq(-3,3, length.out=50)

# L0ara
set.seed(123)
test2 <- cv.l0ara(x = x_norm, y = y_norm, family = "gaussian", lam = lambdas, standardize = FALSE,
                  measure = "mse", nfolds = 10, eps = 1E-7)
plot(test2$lambda, test2$cv.error, type = "l", log = "xy")
abline(v=test2$lam.min, col="red")
min(test2$cv.error) # 0.0002622245 at opt
test2$lam.min # 0.0004714866 = L0Learn lambda / 3.3
test1$fit$lambda[[1]][which.min(test1$cvMeans[[1]])] / test2$lam.min
# there's currently no way to get the model size in function of lambda
# or the model coefficient paths in function of lambda
plot(beta_true,type="l")
lines(coef_scaling*coef(test2)/
      sapply(1:ncol(x_norm), function (i) sd(x_norm[,i])), col="red", type="l") # FIX BUG HERE IN l0ara COEF
sum(coef(test2)!=0) # 30


# L0glm without positivity constraints
# (gives roughly same result as l0ara when using same seed)
set.seed(123)
test4 <- L0glm(y ~ 0 + ., data = data.frame(y = y_norm, x_norm),
               family = gaussian(),
               lambda = lambdas,
               tune.meth = "10-fold", # "trainval"=training and validation set, "IC"=on full data, "5-fold" for nfold CV
               tune.crit = "mse",
               train.vs.val = 0.5,
               nonnegative = FALSE, normalize = FALSE,
               control.iwls = list(maxit = 1, thresh = 1E-7),
               control.l0 = list(maxit = 100, rel.tol = 1E-7, warn = TRUE),
               control.fit = list(maxit = 1), verbose = TRUE)
plot(test4$lambda.tune$lambdas,
     test4$lambda.tune$IC[,"mse"], type = "l", log = "x") # this should be the gold standard with tune.method="trainval" and train.vs.val=0.5
abline(v=test4$lambda.tune$best.lam, col="red")
min(test4$lambda.tune$IC[,"mse"]) # 0.0002622245
test4$lambda.tune$best.lam #  0.0004714866 = L0Learn lambda / 3.3

# 4 with sd=1, 0.04 with sd=0.01
plot(beta_true,type="l")
lines(coef_scaling*coef(test4), col="red", type="l")
sum(coef(test4)!=0) # 189 without pos constraints, 143 with pos constraints

# support size in function of lambda
plot(test4$lambda.tune$lambdas,
     test4$lambda.tune$IC[,"k"], type = "l", log = "x")
# lambda that selects on average 1.33 features:
min(test4$lambda.tune$IC[,"k"])
test4$lambda.tune$lambdas[which.min(abs(min(test4$lambda.tune$IC[,"k"])-test4$lambda.tune$IC[,"k"]))] # 0.01389495
# ca matches L0Learn max lambda /2 :
max((crossprod(x_norm, y_norm)^2)/2)/2

# coefficient path in function of lambda :
matplot(test4$lambda.tune$lambdas, coef_scaling*test4$lambda.tune$coefficients.lam, type="l", log="x")
abline(v=test4$lambda.tune$best.lam, col="red")
# support size in function of lambda :
# PS plot(test4$lambda.tune$lambdas,test4$lambda.tune$IC[,"k"], type = "l", log = "x")
# has a bug in it
plot(test4$lambda.tune$lambdas, rowMeans(test4$lambda.tune$coefficients.lam!=0)*p, type="l", log="x")



# L0glm with positivity constraints + some tests on how to pre-specify lambda based on AIC or BIC
set.seed(123)
test4_pos <- L0glm(y ~ 0 + ., data = data.frame(y = y_norm, x_norm),
               family = gaussian(),
               lambda = lambdas,
               tune.meth = "trainval", # "trainval"=training and validation set, "IC"=on full data, "5-fold" for nfold CV
               tune.crit = "mse",
               train.vs.val = 0.5,
               nonnegative = TRUE, normalize = FALSE,
               control.iwls = list(maxit = 1, thresh = 1E-7),
               control.l0 = list(maxit = 100, rel.tol = 1E-7, warn = TRUE),
               control.fit = list(maxit = 1), verbose = TRUE)
plot(test4_pos$lambda.tune$lambdas,
     test4_pos$lambda.tune$IC[,"mse"], type = "l", log = "xy") # this should be the gold standard with tune.method="trainval" and train.vs.val=0.5
abline(v=test4_pos$lambda.tune$best.lam, col="red")
min(test4_pos$lambda.tune$IC[,"mse"]) # 0.0002622245
test4_pos$lambda.tune$best.lam #  0.0004714866 = L0Learn lambda / 3.3

# 4 with sd=1, 0.04 with sd=0.01
plot(beta_true,type="l")
lines(coef_scaling*coef(test4_pos), col="red", type="l")
sum(coef(test4_pos)!=0) # 189 without pos constraints, 143 with pos constraints

# support size in function of lambda
plot(test4_pos$lambda.tune$lambdas,
     test4_pos$lambda.tune$IC[,"k"], type = "l", log = "x")
# lambda that selects on average 1.33 features:
min(test4_pos$lambda.tune$IC[,"k"])
test4_pos$lambda.tune$lambdas[which.min(abs(min(test4_pos$lambda.tune$IC[,"k"])-test4_pos$lambda.tune$IC[,"k"]))] # 0.01389495
# ca matches L0Learn max lambda /2 :
max((crossprod(x_norm, y_norm)^2)/2)/2

# coefficient path in function of lambda :
matplot(test4_pos$lambda.tune$lambdas, coef_scaling*test4_pos$lambda.tune$coefficients.lam, type="l", log="x")
# support size in function of lambda :
# PS plot(test4$lambda.tune$lambdas,test4$lambda.tune$IC[,"k"], type = "l", log = "x")
# has a bug in it
plot(test4_pos$lambda.tune$lambdas, rowMeans(test4_pos$lambda.tune$coefficients.lam!=0)*p, type="l", log="x")


# some tests

plot(test4$lambda.tune$lambdas, test4$lambda.tune$IC[,"loocv"], type = "l", log = "x")
abline(v=test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"loocv"])], col="red") # these 2 match with tune.meth="trainval" & train.vs.val=0.5
abline(v=test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"aic"])], col="blue")
abline(v=test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"aicc"])], col="blue")
abline(v=test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"hq"])], col="blue")
plot(test4$lambda.tune$lambdas, test4$lambda.tune$IC[,"aic"], type = "l", log = "x")
test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"aic"])] # 4 with sd=1, 0.04 with sd=0.01
abline(v=test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"aic"])], col="red")
abline(v=4*sd^2, col="blue") # this matches more or less with tune.meth="5-fold"
abline(v=2*sd^2, col="orange")
abline(v=0.25*sd^2, col="green4") # this matches more or less with tune.meth="IC" and n>>p
test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"aic"])] # 4
4*sd^2 # 4
0.25*sd^2 # 0.25
test4$lambda.tune$IC[,"aic"][which.min(abs((test4$lambda.tune$lambdas)-(4*sd^2)))] # AIC with preset lambda = -92.19644
test4$lambda.tune$IC[,"aic"][which.min(abs((test4$lambda.tune$lambdas)-(2*sd^2)))]
test4$lambda.tune$IC[,"aic"][which.min(test4$lambda.tune$IC[,"aic"])] # actual optimal AIC = -92.19644 = identical
# lambda that minimizes AIC = 3.98 if sd=1, 0.0398 if sd=0.1
# so this value = sd^2 * factor of 4 of Frommlet & Nuel (proven for orthogonal design,
# though they say its a factor of 1/4 not 4)

plot(test4$lambda.tune$lambdas, test4$lambda.tune$IC[,"bic"], type = "l", log = "x")
test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"bic"])] # 4
4*(log(n)/2)*sd^2 # 10.6
abline(v=test4$lambda.tune$lambdas[which.min(test4$lambda.tune$IC[,"bic"])], col="red")
abline(v=4*(log(n)/2)*sd^2, col="blue")
abline(v=0.25*(log(n)/2)*sd^2, col="green4") # this matches more or less with tune.meth="IC" and n=p
test4$lambda.tune$IC[,"bic"][which.min(abs((test4$lambda.tune$lambdas)-(4*(log(n)/2)*sd^2)))] # BIC with preset lambda = -67.93912
test4$lambda.tune$IC[,"bic"][which.min(test4$lambda.tune$IC[,"bic"])] # actual optimal BIC

# lambda that minimizes BIC = 3.98 if sd=1, 0.0398 if sd=0.1
#  this value should be = sd^2 * ln(n)/2 * factor of 4 of Frommlet & Nuel (proven for orthogonal design)



# bestsubset
# test3 <- bs(x = x, y = y, k = 0:15, intercept = TRUE,
#            form = ifelse(nrow(x) < ncol(x), 2, 1), time.limit = 100, nruns = 50,
#            maxiter = 1000, tol = 1e-07, polish = TRUE, verbose = T)
# This is super slow to compute ... Best k will be set to true k = 10


microbenchmark(
  # L0 penalized regression using L0Learn
  "L0Learn" = {
    L0Learn_fit <- L0Learn.fit(x = x_norm, y = y_norm, penalty="L0", maxSuppSize = ncol(x),
                               nGamma = 0, autoLambda = FALSE, lambdaGrid = list(0.0015593),
                               tol = 1E-7)
  },
  # L0 penalized regression using L0ara
  "L0ara" = {
    L0ara_fit <- l0ara(x = x_norm, y = y_norm, family = "gaussian", lam = 0.001258925,
                       standardize = FALSE, eps = 1E-7)
  },
  # Best subset regression using bestsubset
  "bestsubset" = {
    bs_fit <- bs(x = x_norm, y = y_norm, k = k, intercept = FALSE,
                 form = ifelse(nrow(x) < ncol(x), 2, 1), time.limit = 5, nruns = 50,
                 maxiter = 1000, tol = 1e-7, polish = TRUE, verbose = FALSE)
  },
  # L0 penalized regression using L0glm
  "L0glm" = {
    L0glm_fit <- L0glm(y_norm ~ 0 + ., data = data.frame(y_norm = y_norm, x_norm),
                       family = gaussian(),
                       lambda = 0.0002511886, tune.meth = "none", nonnegative = FALSE,
                       normalize = FALSE,
                       control.iwls = list(maxit = 100, thresh = 1E-7),
                       control.l0 = list(maxit = 100, rel.tol = 1E-7),
                       control.fit = list(maxit = 1), verbose = FALSE)
  },
  times = 1
)
# Note that bestsubset is optimized using the true number of nonzero coefficient
# because tuning it was much to slow. The algorithm check solution using
# Gurobi's mixed integer program solver which is very slow for k = 10, so time
# limit was set to 5 s which dramatically overestimates to true time
# performance of bestsubset

# Check results
df <- data.frame(coef.L0Learn = coef_scaling*as.numeric(L0Learn_fit$beta[[1]]),
                 coef.L0ara = coef_scaling*L0ara_fit$beta,
                 coef.bestsubset = coef_scaling*as.vector(bs_fit$beta),
                 coef.L0glm = coef_scaling*coef(L0glm_fit),
                 coef.true = beta)
all(df[(k+1):p,] == 0)
# No false positives !
abs(df$coef.L0Learn - df$coef.bestsubset)[1:k]
abs(df$coef.L0glm - df$coef.L0ara)[1:k]
abs(df$coef.L0glm - df$coef.L0Learn)[1:k]

data <- data.frame(y = unlist(df[1:k,]),
                   x = rep(1:k, ncol(df[1:k,])),
                   type = rep(c("L0Learn", "L0ara", "bestsubset", "L0glm", "true"), each = k))
pl <- ggplot(data = data, aes(x = x, y = y, color = type)) +
  geom_point() + geom_line() +
  ggtitle("Compare true coefficients with coefficient estimated \nusing bestsubset, L0ara, L0glm, L0Learn") +
  ylab("Estimate") + xlab("Index") +
  scale_colour_manual(name = "Algorithm",
                      values = c(L0Learn = "red3", L0ara = "orange2",
                                 bestsubset = "purple", L0glm = "green4", true = "grey40"),
                      labels = c(L0Learn = "L0Learn", L0ara = "L0ara",
                                 bestsubset = "bestsubset", L0glm = "L0glm", true = "True"))
graph2ppt(pl, file = "Github/graphs", scaling = 50, append = TRUE)

# Conclusion
# All algorithms find the correct set of coefficients
# The solution with L0Learn is very similar to the solution found with bestsubset
# (up to 5E-4). The L0glm and L0ara find almost the same solution (up to 1E-8).
# However, there is a noticeable difference between L0glm (and hence L0ara) and
# bestsubset (and hence L0Learn) and this difference seems to be systematic across
# coefficients.





####---- COMPARE L0 PENALTY WITH LASSO, MCP, OR SCAD PENALTY ----####


library(ncvreg)

# From ?L0Learn.fit examples
# Generate synthetic data for this example
n <- 200
p <- 500
k <- 10
data <- GenSynthetic(n = n, p = p, k = k, seed = 123)
beta <-  c(rep(1, k), rep(0, p - k))
x <- data$X
y <- data$y

microbenchmark(
  "lasso" = {
    lasso_fit <- ncvreg(X = x, y = y, family = "gaussian", penalty = "lasso",
                        alpha = 1, lambda = 0.1278381, eps = 1e-7)
  },
  "MCP" = {
    mcp_fit <- ncvreg(X = x, y = y, family = "gaussian", penalty = "MCP",
                      alpha = 1, lambda = 0.1678555, eps = 1e-7)
  },
  "SCAD" = {
    scad_fit <- ncvreg(X = x, y = y, family = "gaussian", penalty = "SCAD",
                       alpha = 1, lambda = 0.1442869, eps = 1e-7)
  },
  "L0" = {
    L0glm_fit <- L0glm(y ~ 1 + ., data = data.frame(y = y, x),
                       family = gaussian(),
                       lambda = 2.5, tune.meth = "none", nonnegative = FALSE,
                       control.iwls = list(maxit = 100, thresh = 1E-7),
                       control.l0 = list(maxit = 100, rel.tol = 1E-7),
                       control.fit = list(maxit = 1), verbose = FALSE)
  },
  times = 5
)

# Check results
df <- data.frame(coef.lasso = coef(lasso_fit)[-1], # remove intercept
                 coef.mcp = coef(mcp_fit)[-1],
                 coef.scad = coef(scad_fit)[-1],
                 coef.L0 = coef(L0glm_fit)[-1],
                 coef.true = beta)
FP <- colSums(df[(k+1):p,] != 0)
TP <- colSums(df[1:k,] != 0)
FN <- colSums(df[1:k,] == 0)
TN <- colSums(df[(k+1):p,] == 0)
TP/(TP + FN) # sensitivity
TN/(TN + FP) # specificity

# Plot results
data <- data.frame(y = unlist(df),
                   x = rep(1:p, ncol(df)),
                   type = rep(c("lasso", "mcp", "scad", "L0", "true"), each = p))
nz <- (1:p) %in% 2:k
pl <- ggplot(data = data[nz,], aes(x = x, y = y, color = type)) +
  geom_point() + geom_line() +
  ggtitle("Compare true nonzero coefficients with coefficient estimated \nusing lasso, MCP, SCAD, or L0 penalties") +
  ylab("Estimate") + xlab("Index") +
  scale_colour_manual(name = "Algorithm",
                      values = c(lasso = "red3", mcp = "orange2", scad = "purple", L0 = "green4", true = "grey40"),
                      labels = c(lasso = "lasso", mcp = "MCP", scad = "SCAD",  L0 = "L0glm", true = "True"))
graph2ppt(pl, file = "Github/graphs", scaling = 50, append = TRUE)


# TODO delete
# Opimize lambda
lasso_cv <- cv.ncvreg(X = x, y = y, family = "gaussian", penalty = "lasso",
                      alpha = 1, eps = 1e-9, nfolds = 3, seed = 123)
plot(x = lasso_cv$lambda, y = lasso_cv$cve, log = "xy", type = "l")
lasso_cv$lambda.min # 0.1278381
mcp_cv <- cv.ncvreg(X = x, y = y, family = "gaussian", penalty = "MCP",
                    alpha = 1, eps = 1e-9, nfolds = 3, seed = 123)
plot(x = mcp_cv$lambda, y = mcp_cv$cve, log = "xy", type = "l")
mcp_cv$lambda.min # 0.1678555
scad_cv <- cv.ncvreg(X = x, y = y, family = "gaussian", penalty = "SCAD",
                     alpha = 1, eps = 1e-9, nfolds = 3, seed = 123)
plot(x = scad_cv$lambda, y = scad_cv$cve, log = "xy", type = "l")
scad_cv$lambda.min # 0.1442869
L0glm_cv <- L0glm(y ~ 1 + ., data = data.frame(y = y, x),
                  family = gaussian(),
                  lambda = 10^seq(-1, 1, length.out = 100), tune.meth = "3-fold",
                  tune.crit = "rss", nonnegative = FALSE, seed = 123,
                  control.iwls = list(maxit = 100, thresh = 1E-4),
                  control.l0 = list(maxit = 100, rel.tol = 1E-7, warn = TRUE),
                  control.fit = list(maxit = 1), verbose = TRUE)
plot(x = L0glm_cv$lambda.tune$lambdas, y = L0glm_cv$lambda.tune$IC[,"rss"], log = "xy", type = "l")
L0glm_cv$lambda.tune$best.lam # 2.25702







#### EXTRA TESTS ####



#### TEST NEED FOR STANDARDIZATION ####


# Simulate some data
set.seed(1234)
n = 200
npeaks = 20
sim <- simulate_spike_train(Plot = TRUE, n = n)
x <- sim$X
scales <- runif(n, min = -1E2, max = 1E2)
x <- sweep(x, 2, scales, "*")
beta <- runif(n, min = -1, max = 1)
beta[sample(1:length(beta), size = length(beta)-25)] <- 0
y0 <- x %*% beta
y <- y0  + rnorm(n, sd = 2.5)
plot(y)
lines(y0)
abline(h = 0)

par(mfrow = c(3,1))
plot(y, main = "True model")
matlines(sweep(x, 2, beta, "*"), lty = 1, type = "l", col = "orange2")
# No normalization
L0glm.out <- L0glm(formula = y ~ 0 + ., data = data.frame(y = y, x),
                   family = gaussian(identity),
                   lambda = 1000, tune.meth = "none", nonnegative = FALSE,
                   control.iwls = control.iwls.gen(maxit = 1),
                   control.l0 = control.l0.gen(),
                   control.fit = control.fit.gen())
plot(y, main = "No normalization")
matlines(sweep(x, 2, L0glm.out$coefficients, "*"), lty = 1, type = "l", col = "orange2")
# With normalization
x <- sweep(x, 2, apply(x, 2, norm, type = "2"), "/")
L0glm.out <- L0glm(formula = y ~ 0 + ., data = data.frame(y = y, x),
                   family = gaussian(identity),
                   lambda = 1, tune.meth = "none", nonnegative = FALSE,
                   control.iwls = control.iwls.gen(maxit = 1),
                   control.l0 = control.l0.gen(),
                   control.fit = control.fit.gen())
plot(y, main = "With normalization")
matlines(sweep(x, 2, L0glm.out$coefficients, "*"), lty = 1, type = "l", col = "orange2")



#### TEST L0GLM AGAINST GLM ####

# Compare L0glm against stats::glm
X.sub <- X[, seq(1,ncol(X), by = 5)] # Subsampling needed otherwise
# gaussian: identity, log and inverse
# binomial: logit, probit, cauchit, log, and cloglog
# Gamma: inverse, identity and log
# poisson: log, identity, and sqrt
# inverse.gaussian: 1/mu^2, inverse, identity and log.
# TODO test the qusi families, quasi: logit, probit, cloglog, identity, inverse, log, 1/mu^2 and sqrt
fam <- poisson()
microbenchmark(cpglm <- pen.nnglm(X = X.sub, y = y, family = fam, lambda = 0, maxit.l0 = 1, maxit.iwnnls = 25, cv = "none", constr = "none"),
               glm <- glm.fit(x = X.sub, y = y, family = fam, control = list(maxit = 25, epsilon = 1E-8), intercept = FALSE),
               times = 50)
cbind(cpglm = cpglm$coefficients, glm = glm$coefficients, diff.abs = abs(cpglm$coefficients - glm$coefficients))

par(mfrow = c(2,1))
plot(cpglm$coefficients, type = "l", col = 4)
lines(glm$coefficients, col = 2)
plot(y, pch = 16, cex = 0.5, col = "grey")
lines(cpglm$fitted.values, col = 4)
lines(glm$fitted.values, col = 2)

par(mfrow = c(1,1))
plot(y, pch = 16, col = "grey")
lines(y_nonoise, col = 2)
lines(cpglm$fitted.values, col = "orange2")






# Simulate data
get.data.poisson.nn <- function(n = 200, npeaks = 20,seed = 123,
                                peakhrange = c(10,1E3), Plot = F,
                                envir = .GlobalEnv){
  set.seed(seed)
  x = 1:n
  # unkown peak locations
  u = sample(x, npeaks, replace=FALSE)
  # unknown peak heights
  h = 10^runif(npeaks, min=log10(min(peakhrange)), max=log10(max(peakhrange)))
  # locations of spikes of simulated spike train, which are assumed to be unknown
  # here, and which needs to be estimated from the measured total signal
  a = rep(0, n)
  a[u] = h
  # peak shape function
  gauspeak = function(x, u, w, h=1) h*exp(((x-u)^2)/(-2*(w^2)))
  # banded matrix with peak shape measured beforehand
  X = do.call(cbind, lapply(1:n, function (u) gauspeak(x, u=u, w=5, h=1) ))
  # X = Matrix(X, sparse = T)
  # colnames(X) <- paste("Var", 1:ncol(X))
  # noiseless simulated signal = linear convolution of spike train with peak shape function
  y_nonoise = as.vector(X %*% a)
  # simulated signal with random poisson noise
  y <- rpois(n, y_nonoise)

  # Plot the data
  if(Plot){
    par(mfrow=c(1,1))
    plot(y, type = "l", ylab = "Signal", xlab = "x",
         main = "Simulated spike train (red) to be estimated given known blur kernel & with Poisson noise")
    lines(a, type = "h", col = "red")
  }
  l <- mget(ls(environment()))
  invisible(list2env(x = l[!names(l) %in% c("envir", "Plot", "gauspeak", "seed")], envir = envir))
}

################################################################
#### COMPARE OUR PACKAGE VS STATS GLM FOR NON PENALIZED GLM ####
################################################################


sim <- simulate_spike_train(Plot = TRUE, n = n)
X <- sim$X
y <- sim$y
# Test our package vs glm
X.sub <- X[,seq(1,ncol(X), by = 5)]
# gaussian: identity, log and inverse
# binomial: logit, probit, cauchit, log, and cloglog
# Gamma: inverse, identity and log
# poisson: log, identity, and sqrt
# inverse.gaussian: 1/mu^2, inverse, identity and log.
# TODO test the qusi families, quasi: logit, probit, cloglog, identity, inverse, log, 1/mu^2 and sqrt
fam <- poisson()
microbenchmark(cpglm <- L0glm(y ~ 0 + ., data = data.frame(y = y, X.sub), family = fam, lambda = 0,
                              control.l0 = control.l0.gen(maxit = 1),
                              control.iwls = control.iwls.gen(maxit = 1),
                              nonnegative = FALSE, tune.meth = "none", verbose = FALSE),
               glm <- glm.fit(x = X.sub, y = y, family = fam, control = list(maxit = 25, epsilon = 1E-8), intercept = FALSE),
               times = 50)
cbind(cpglm = cpglm$coefficients, glm = glm$coefficients, diff.abs = abs(cpglm$coefficients - glm$coefficients))

par(mfrow = c(2,1))
plot(cpglm$coefficients, type = "l", col = 4)
lines(glm$coefficients, col = 2)
plot(y, pch = 16, cex = 0.5, col = "grey")
lines(cpglm$fitted.values, col = 4)
lines(glm$fitted.values, col = 2)

par(mfrow = c(1,1))
plot(y, pch = 16, col = "grey")
lines(y_nonoise, col = 2)
lines(cpglm$fitted.values, col = "orange2")


# Binomial case

set.seed(123)
n <- 200
ncov <- 25
beta.t <- rep(0, n)
nz <- sample(1:n, ncov)
beta.t[nz] <- runif(ncov, min = -10, max = 10)
eta.true <- X %*% beta.t
y.true <- binomial()$linkinv(eta.true)
y <- as.numeric(y.true >= 0.5)
plot(y, pch = 15)
lines(y.true, col = 2)

# Test our package vs glm
X.sub <- X[,seq(1,ncol(X), by = 5)]
# gaussian: identity, log and inverse
# binomial: logit, probit, cauchit, log, and cloglog
# Gamma: inverse, identity and log
# poisson: log, identity, and sqrt
# inverse.gaussian: 1/mu^2, inverse, identity and log.
# TODO test the qusi families, quasi: logit, probit, cloglog, identity, inverse, log, 1/mu^2 and sqrt
fam <- binomial()
microbenchmark(cpglm <- pen.nnglm(X = X.sub, y = y, family = fam, lambda = 0, maxit.l0 = 1, maxit.iwnnls = 25, cv = "none", constr = "none"),
               glm <- glm.fit(x = X.sub, y = y, family = fam, control = list(maxit = 25, epsilon = 1E-8), intercept = FALSE),
               times = 2)
cbind(cpglm = cpglm$coefficients, glm = glm$coefficients, diff.abs = abs(cpglm$coefficients - glm$coefficients))
# Same results
par(mfrow = c(2,1))
plot(cpglm$coefficients, type = "l", col = 4)
lines(glm$coefficients, col = 2)
plot(y, pch = 16, cex = 0.5, col = "grey")
lines(cpglm$fitted.values, col = 4)
lines(glm$fitted.values, col = 2)

# Test to extract only coefficients that are positively correlated with
# increased risk = nonegativity constrained
cpglm <- pen.nnglm(X = X, y = y, family = fam, lambda = 10^seq(-5,2), maxit.l0 = 25, maxit.iwnnls = 25, cv = "loocv", constr = "nonneg")
plot(eta.true, type = "l")
lines(beta.t, col = 2, type = "h")
lines(X %*% cpglm$coefficients, col = "green4")
lines(cpglm$coefficients, col = "green4", type = "h")


############################################################
#### COMPARE OUR PACKAGE VS OTHER PACKAGE FOR RIDGE GLM ####
############################################################


get.data.poisson.nn()
X.sub <- X[,seq(1,ncol(X), by = 5)]
cpglm <- L0glm(X = X.sub, y = y, family = gaussian(identity),
               control.l0 = control.l0(maxit = 1),
               control.fit = control.fit(constr = "none"),
               lambda = 1E-3, cv = "none")
library(ridge)
ridge.test <- linearRidge(y ~ 0 + ., data = data.frame(y = y, as.matrix(X.sub)),
                          lambda = 1E-3, nPCs = NULL, scaling = "none")
library(glmnet)
glmnet.fit <- glmnet(x = X.sub, y = y, family = "gaussian", lambda = 1E-3,
                     alpha = 0, standardize = F, intercept = F)
library(penalized)
pen.fit <- penalized(response = y, penalized = as.matrix(X.sub),
                     unpenalized = ~ 0, lambda1 = 0, lambda2 = 1E-3,
                     model = "linear", standardize = F)

plot(cpglm$coefficients, type = "l")
lines(ridge.test$coef, col = 2)
lines(glmnet.fit$beta, col = 4)
lines(pen.fit@penalized, col = "orange2")


cpglm <- pen.nnglm(X = X, y = y, family = gaussian(identity), constr = "none",
                   lambda = 200, maxit.l0 = 25, cv = "none")
glmnet.fit <- glmnet(x = X, y = y, family = "gaussian", lambda = 10, thresh = 1E-16,
                     alpha = 1, standardize = F, intercept = F)
par(mfrow = c(2,1))
plot.nnglm.benchmark(x = x, y = y, fit = nnglm_pois_L0, a.true = a,
                     main="Ground truth vs L0 penalized NNGLM estimates")
glmnet.fit$coefficients = as.vector(glmnet.fit$beta)
glmnet.fit$fitted.values = X %*% glmnet.fit$coefficients
plot_L0glm_benchmark(x = x, y = y, fit = glmnet.fit, a.true = a,
                     main="Ground truth vs L0 penalized NNGLM estimates")


################################################################
#### COMPARE OUR PACKAGE VS OTHER PACKAGE FOR L0 REGRESSION ####
################################################################

library(L0Learn)
# Simulate some data
get.data.poisson.nn()
# L0Learn
system.time(L0fit <- L0Learn.fit(x=X, y=y, penalty="L0", algorithm="CD", nLambda=50, maxSuppSize = 200))
# Note algortihm == "CDPSI"should give better results but this is not the case here
# L0glm
system.time(L0glmfit <- L0glm(X = X, y = y, family = gaussian(identity), intercept = F,
                              lambda = 10^seq(-1, 4, length.out = 51), cv = "none",
                              control.fit = control.fit(constr = "none"),
                              control.iwls = control.iwls(rel.tol = 1E-4, maxit = 100, thresh = 1E-1),
                              control.l0 = control.l0(rel.tol = 1E-4, maxit = 100, delta = 1E-3), verbose = F))
# check coefficient trace
matplot(x = unlist(L0fit$lambda), y = t(abs(as.matrix(L0fit$beta[[1]]))), type = "l", log = "xy")
matplot(x = colnames(L0glmfit$coefficients), y = t(abs(L0glmfit$coefficients)), type = "l", log = "xy")

# Check best solution
par(mfrow = c(3,1))
# True
plot(y, main = "True")
lines(X %*% a, col = 2)
matlines(sweep(X, 2, a, "*"), lty = 1, col = "green4")
# L0Learn
good <- 10
plot(y, main = "L0Learn")
lines(X %*% L0fit$beta[[1]][,good], col = 2)
matlines(sweep(X, 2, L0fit$beta[[1]][,good], "*"), lty = 1, col = "green4")
# L0glm
good <- 31
plot(y, main = "L0glm")
lines(L0glmfit$fitted.values[,good], col = 2)
matlines(sweep(X, 2, L0glmfit$coefficients[,good], "*"), lty = 1, col = "green4")


##################################
#### ISSUE WITH SPARSE MATRIX ####
##################################


X.test <- matrix(runif(1000), ncol = 10)
X.test[sample(1:1000, 500)] <- 0
X.test <- Matrix(X.test, sparse = T)
coef.test <- runif(10)

t1 <- X.test %*% coef.test
t2 <- as.matrix(X.test) %*% coef.test
sum(abs(t1-t2))
sum(abs(X.test-as.matrix(X.test)))


###########################################################
#### TRY MATCHING BEST LAMBDA TO THEORETICAL IC LAMBDA ####
###########################################################


get.data.poisson.nn()
system.time(nnglm_pois_L0 <- L0glm(X = X, y = y, family = poisson(identity), intercept = FALSE,
                                   nonnegative = TRUE,
                                   tune.crit = "bic", tune.meth = "IC",
                                   lambda = 10^seq(4,6, length.out = 21), # Use arbitrary sequence
                                   control.fit = list(),
                                   control.iwls = list(thresh = 1E-7, rel.tol = 1E-4),
                                   control.l0 = list(maxit = 50)))
matplot(nnglm_pois_L0$lambda.tune$coefficients.lam, type = "l", log = "", lty = 1)
par(mfrow = c(2,2))
for(i in 1:ncol(nnglm_pois_L0$lam.opt$IC)){
  sub <- 1:18
  matplot(x = nnglm_pois_L0$lam.opt$lambdas[sub], y = nnglm_pois_L0$lam.opt$IC[sub,i], type = "l", main = colnames(nnglm_pois_L0$lam.opt$IC)[i], log = "x")
  abline(v = nnglm_pois_L0$lam.opt$best.lam)
  abline(v = nnglm_pois_L0$lam.opt$lambdas[which.min(nnglm_pois_L0$lam.opt$IC[,i])], col = "red")

}
plot.nnglm.benchmark(x = x, y = y, fit = nnglm_pois_L0, a.true = a,
                     main="Ground truth vs L0 penalized NNGLM estimates (BIC on full data)")


###########################################
#### TEST BEST LAMBDA SELECTION METHOD ####
###########################################

# Common constants to all simulations
n = 200
x = 1:n
# unknown peak heights
peakhrange = c(10,1E3)
# peak shape function
gauspeak = function(x, u, w, h=1) h*exp(((x-u)^2)/(-2*(w^2)))
# banded matrix with peak shape measured beforehand
X = do.call(cbind, lapply(1:n, function (u) gauspeak(x, u=u, w=5, h=1) ))

# Fitting
fam <- poisson("identity")
lambdas <- 10^seq(-8,2)

# Range of nonzero coefficients for simulation
K.range <- 1:25
# Generate simulated data for every K
set.seed(123)
res <- list()
for(K in K.range){
  # Simulate data
  # Unkown peak coefficients
  u = sample(x, K, replace=FALSE)
  h = 10^runif(K, min=log10(min(peakhrange)), max=log10(max(peakhrange)))
  a = rep(0, n)
  a[u] = h
  # Response variable
  y_nonoise = as.vector(X %*% a)
  y <- rpois(n, y_nonoise)

  # Split data in training (90%) and validation (10%)
  train <- sort(sample(1:n, n*0.9))
  X.train <- X[train,]
  y.train <- y[train]
  X.test <- X[-train,]
  y.test <- y[-train]

  # Fit data after optimizing lambda for L0 penalty
  # BIC
  print("BIC")
  cpglm <- pen.nnglm(X = X.train, y = y.train, family = fam, constr = "none",
                     lambda = lambdas, maxit.l0 = 1, cv = "full.data", cv.crit = "bic", inference = TRUE)
  glm.fit(x = X.train, y = y.train, family = fam, intercept = F)

  y.hat.test <- X.test %*% cpglm$coefficients
  BIC.mse <- sum((y.test - y.hat.test)^2)/length(y.test)
  BIC.p0 <- sum(cpglm$coefficients != 0)
  # AIC
  print("AIC")
  cpglm <- pen.nnglm(X = X.train, y = y.train, family = fam, constr = "none",
                     lambda = lambdas, maxit.l0 = 1, cv = "full.data", cv.crit = "aic", inference = TRUE)
  y.hat.test <- X.test %*% cpglm$coefficients
  AIC.mse <- sum((y.test - y.hat.test)^2)/length(y.test)
  AIC.p0 <- sum(cpglm$coefficients != 0)
  # LOOCV
  print("LOOCV")
  cpglm <- pen.nnglm(X = X.train, y = y.train, family = fam, constr = "none",
                     lambda = lambdas, maxit.l0 = 1, cv = "loocv", inference = TRUE)
  y.hat.test <- X.test %*% cpglm$coefficients
  LOOCV.mse <- sum((y.test - y.hat.test)^2)/length(y.test)
  LOOCV.p0 <- sum(cpglm$coefficients != 0)
  # loglik
  print("loglik")
  cpglm <- pen.nnglm(X = X.train, y = y.train, family = fam, constr = "none",
                     lambda = lambdas, maxit.l0 = 1, cv = "full.data", cv.crit = "loglik", inference = TRUE)
  y.hat.test <- X.test %*% cpglm$coefficients
  loglik.mse <- sum((y.test - y.hat.test)^2)/length(y.test)
  loglik.p0 <- sum(cpglm$coefficients != 0)

  # Store results
  res[K] <- list(MSE = c(BIC = BIC.mse, AIC = AIC.mse, LOOCV = LOOCV.mse, loglik = loglik.mse),
                 p0 = c(BIC = BIC.p0, AIC = AIC.p0, LOOCV = LOOCV.p0, loglik = loglik.p0))

  print.progress(K, max(K.range))
  cat("\n")
}
#




########################################
#### FIND OPTIMAL LAMBDA WITHOUT CV ####
########################################

# See Theorem 1 in Frommlet et al 2016.

# 1. Generate orthogonal matrix
library(pracma)
set.seed(1234)
n <- 200
p <- 200
X <- randortho(n)
X <- X * sqrt(n)
X <- X[,1:p]

# 2. Simulate coefficients and generate data with gaussian error structure
beta <- rep(0, p)
s <- 5
beta[1:s] <- 1
y0 <- X %*% beta
sigma2 <- 10
y <- y0 + rnorm(mean = 0, sd = sigma2, n = n)

# 3. Tune lambda
# theoetically, BIC should be optimized when lambda = log(n)/4
lams <- 10^(log10(log(n)/4) + seq(-4, 2, length.out = 150))
L0glm_fit <- L0glm(X = X, y = y, family = gaussian(identity),
                   intercept = FALSE, lambda = lams, # Use arbitrary sequence
                   tune.crit = "bic", tune.meth = "IC", seed = 123,
                   nonnegative = F,
                   control.fit = list(),
                   control.iwls = list(rel.tol = 1E-4, maxit = 50, thres = 1E-6),
                   control.l0 = list(rel.tol = 1E-4, maxit = 100, delta = 1E-6))

# 4. Check results
k <- rowSums(L0glm_fit$lambda.tune$coefficients.lam!=0)
coef.dif <- apply(L0glm_fit$lambda.tune$coefficients.lam, 1, function(x) sum(abs(x - beta)) )
rss <- L0glm_fit$lambda.tune$IC[,"rss"]
min2ll <- -2*L0glm_fit$lambda.tune$IC[,"loglik"]
par(mfrow = c(3,2))
for(crit in c("aic", "bic")){
  # AIC and BIC based on log likelihood
  # plot(x = L0glm_fit$lambda.tune$lambdas, y = L0glm_fit$lambda.tune$IC[,crit],
  #      log = "x", type = "l", main = crit)

  # Custom IC as defined in Frommlet et Nuel where BIC = 1/sima2*RSS + k * log(n)
  # sigma2 is known from the data generation (not the case in real world problems)
  plot(x = L0glm_fit$lambda.tune$lambdas, y = rss/sigma2 + k * ifelse(crit == "aic", 2, log(n)),
       log = "x", type = "l", main = crit)

  abline(v = lams[which.min(L0glm_fit$lambda.tune$IC[,crit])], col = 2)
  abline(v = 1/4 * ifelse(crit == "aic", 2, log(n)))
}
sigs <- apply(L0glm_fit$lambda.tune$coefficients.lam,  1, function(beta) sum((y - X %*% beta)^2))
sigs <- sigs/(n-k)
plot(x = L0glm_fit$lambda.tune$lambdas, y = rss/sigs, log = "x", type = "l", col = 4, main = "RSS/sig")
plot(x = L0glm_fit$lambda.tune$lambdas, y = min2ll, log = "x", type = "l", col = 4, main = "-2logLik")
plot(x = L0glm_fit$lambda.tune$lambdas, y = k, main = "Number of non zero coefficients", type = "l", log = "x")
abline(h = s, lty=2)
plot(x = L0glm_fit$lambda.tune$lambdas, y = coef.dif, main = "Coefficient distance to true coefficients", type = "l", log = "x", ylim = c(0, max(coef.dif)))
abline(h = 0, lty = 2)
abline(v = L0glm_fit$lambda.tune$lambdas[which.min(coef.dif)], col = 2)

matplot(x = L0glm_fit$lambda.tune$lambdas, y = L0glm_fit$lambda.tune$coefficients.lam, type = "l", lty = 1, log = "x")

par(mfrow = c(3,2))
plot(x = L0glm_fit$lambda.tune$lambdas, y = min2ll, log = "x", type = "l", col = 4, main = "-2logLik")
plot(x = L0glm_fit$lambda.tune$lambdas, y = rss, log = "x", type = "l", col = 4, main = "RSS")
plot(x = L0glm_fit$lambda.tune$lambdas, y = min2ll+2*k, log = "x", type = "l", col = 4, main = "AIC.ll")
abline(v = 0.5)
abline(v = L0glm_fit$lambda.tune$lambdas[which.min(min2ll+2*k)], col = 2)
plot(x = L0glm_fit$lambda.tune$lambdas, y = min2ll+log(n)*k, log = "x", type = "l", col = 4, main = "BIC.ll")
abline(v = log(n)/4)
abline(v = L0glm_fit$lambda.tune$lambdas[which.min(min2ll+log(n)*k)], col = 2)
plot(x = L0glm_fit$lambda.tune$lambdas, y = rss+2*k, log = "x", type = "l", col = 4, main = "AIC.rss")
abline(v = 0.5)
abline(v = L0glm_fit$lambda.tune$lambdas[which.min(rss+2*k)], col = 2)
plot(x = L0glm_fit$lambda.tune$lambdas, y = rss+log(n)*k, log = "x", type = "l", col = 4, main = "BIC.rss")
abline(v = log(n)/4)
abline(v = L0glm_fit$lambda.tune$lambdas[which.min(rss+log(n)*k)], col = 2)


# Poisson family

get.data.poisson.nn()
lams <- 10^(log10(log(n)/4) + seq(-2, 2, length.out = 50))
L0glm.pois <- L0glm(X = X, y = y, family = poisson(identity),
                    lambda = lams, intercept = FALSE,
                    tune.crit = "bic", tune.meth = "IC", seed = 123,
                    control.fit = control.fit(constr = "nonneg"),
                    control.iwls = control.iwls(rel.tol = 1E-4, maxit = 50, thres = 1E-6),
                    control.l0 = control.l0(rel.tol = 1E-4, maxit = 100, delta = 1E-6))

k <- rowSums(L0glm.pois$lambda.tune$coefficients.lam!=0)
coef.dif <- apply(L0glm.pois$lambda.tune$coefficients.lam, 1, function(x) sum(abs(x - beta)) )
rss <- L0glm.pois$lambda.tune$IC[,"rss"]
min2ll <- -2*L0glm.pois$lambda.tune$IC[,"loglik"]
par(mfrow = c(3,2))
for(crit in c("aic", "bic")){
  plot(x = L0glm.pois$lambda.tune$lambdas, y = L0glm.pois$lambda.tune$IC[,crit],
       log = "x", type = "l", main = crit)
  abline(v = lams[which.min(L0glm.pois$lambda.tune$IC[,crit])], col = 2)
  abline(v = 1/4 * ifelse(crit == "aic", 2, log(n)))
}
plot(x = L0glm.pois$lambda.tune$lambdas, y = (rss), log = "x", type = "l", col = 4, main = "RSS")
plot(x = L0glm.pois$lambda.tune$lambdas, y = min2ll, log = "x", type = "l", col = 4, main = "-2logLik")
plot(x = L0glm.pois$lambda.tune$lambdas, y = k, main = "Number of non zero coefficients", type = "l", log = "x")
abline(h = npeaks, lty=2)
plot(x = L0glm.pois$lambda.tune$lambdas, y = coef.dif, main = "Coefficient distance to true coefficients", type = "l", log = "x", ylim = c(0, max(coef.dif)))
abline(h = 0, lty = 2)
abline(v = L0glm.pois$lambda.tune$lambdas[which.min(coef.dif)], col = 2)


# Find maximum lambda
# https://stats.stackexchange.com/questions/416144/minimum-and-maximum-regularization-in-l0-pseudonorm-penalized-regression/417080#417080

library(L0Learn)
# Simulate some data
data <- GenSynthetic(n=100,p=500,k=30,seed=123)
X = data$X
y = data$y
# get.data.poisson.nn()
# make L0 penalized fit:
lams <- list()
lams[[1]] <- 10^seq(-9,9 ,length.out = 101)
L0fit <- L0Learn.fit(x=X, y=y, penalty="L0", algorithm="CD", maxSuppSize = 100, autoLambda = F, lambdaGrid = lams)
# Does not work...
unlist(L0fit$lambda)[unlist(L0fit$suppSize)==0][1] # = 0.0618124
unlist(L0fit$lambda)[unlist(L0fit$suppSize)==max(unlist(L0fit$suppSize))][1] # = 6.5916e-09

L0glm.res <- L0glm(X = X, y = y, family = gaussian(identity), start = NULL,
                   no.pen = 0, intercept = F, lambda = lams[[1]], control.l0 = control.l0(maxit = 100, rel.tol = 1E-9),
                   control.fit = control.fit(constr = "none"), control.iwls = control.iwls(), tune.meth = "none")
k <- colSums(L0glm.res$coefficients != 0)

max.th <- 1/2 * max(1/diag(crossprod(X, X)) * crossprod(X, y)^2)
max.emp <- lams[[1]][min(which(k == 0))]
max.th
max.emp
max.th/max.emp

# CCL: there is not an exact match, especially for our shifted gaussian case where
# the fold difference can be up to +/-1000. However the theoretically value seems
# to be sysematically higher than the empirical max lambdda. So it can still be
# use as a threshold to avoid testing too large lambdas !

plot(L0glm.res$coefficients[,86])






############################################################
#### TEST IF AR AND IWLS CAN BE OPTIIZED SIMULTANEOUSLY ####
############################################################

# Simulated poisson data
get.data.poisson.nn(Plot = T)

# Nested update
system.time(L0glm.nu <- L0glm(X = X, y = y, family = poisson(identity),
                              intercept = FALSE, lambda = 1,
                              control.fit = control.fit(constr = "nonneg"),
                              control.iwls = control.iwls(rel.tol = 1E-4, maxit = 50, thresh = 1E-4),
                              control.l0 = control.l0(rel.tol = 1E-4, maxit = 100)))
plot_L0glm_benchmark(x = x, y = y, fit = L0glm.nu, a.true = a,
                     main="Nested updates")


# Simultaneous update
system.time(L0glm.su <- L0glm(X = X, y = y, family = poisson(identity),
                              intercept = FALSE, lambda = 1,
                              control.fit = control.fit(constr = "nonneg"),
                              control.iwls = control.iwls(rel.tol = 1E-4, maxit = 1, thresh = 1E-4),
                              control.l0 = control.l0(rel.tol = 1E-4, maxit = 100)))
plot_L0glm_benchmark(x = x, y = y, fit = L0glm.su, a.true = a,
                     main="Nested updates")

# CCL: updating both AR and IWLS simultaneously lead to similar fit (maybe evn better)
# than the nested version and is +/- 5x faster (+/-5 x less iterations)

#################################################################
#### CHECK STABILITY OF THE ESTIMATES OVER A SERIE OF LAMBDA ####
#################################################################

# Simulated poisson data
get.data.poisson.nn(Plot = T)

lams <- 10^seq(-2, 2, length.out = 101)
L0glm <- L0glm(X = X, y = y, family = poisson(identity),
               intercept = FALSE, lambda = lams,
               cv = "full.data", cv.crit = "bic",
               control.fit = control.fit(constr = "nonneg"),
               control.iwls = control.iwls(rel.tol = 1E-4, maxit = 1, thresh = 1E-4),
               control.l0 = control.l0(rel.tol = 1E-4, maxit = 100))
plot_L0glm_benchmark(x = x, y = y, fit = L0glm, a.true = a,
                     main="Nested updates")

matplot(L0glm$lambda.tune$coefficients.lam, type = "l", lty = 1)



############################
#### TEST ON LARGE DATA ####
############################


# Use a random GC-MS sample
source("D:/Documents/Dropbox/christophe/_tests deconvolution scripts/utils-0.0.2.R")
M <- read.cdf("D:/Documents/beer_chromatograms_miguel/20170129/54085015.cdf")
y <- rowSums(M)
n <- length(y)
peak.widths <- fit.average.peak.width(y = y,win.size = 300,
                                      peak.width.range = c(1,20))
peak.width.fun <- peak.widths$peak.width.fun
peak.widths <- peak.widths$peak.width.fun$fitted.values
peak.widths.preds <- predict(peak.width.fun, newdata = list(x = 1:length(y)))
X <- build.banded.gaus.mat(w = peak.widths.preds, tol = 1E-5)

deconv.fit <- L0glm(X = X, y = y, family = poisson(identity), intercept = F,
                    lambda = 1,
                    control.l0 = control.l0(maxit = 25),
                    control.iwls = control.iwls(thresh = 1, maxit = 1),
                    control.fit = control.fit(maxit = 25, block.size = 100, constr = "nonneg"))

xlims <- c(4600, 5100)
plot(x = 1:n, y = y, xlim = xlims, type = "l", log = "y")
lines(x = 1:n, y = deconv.fit$fitted.values, col = "orange2")
matlines(x = 1:n, y = sweep(X[,xlims[1]:xlims[2]], 2, deconv.fit$coefficients[xlims[1]:xlims[2]], "*"), lty = 1, col = "green4")

x <- 1:n
bas <- bs(x = 1:n, df = 15+3, degree = 3)
class(bas) <- "matrix"
X.full <- cbind(bas, X)

deconv.fit <- L0glm(X = X.full, y = y, family = poisson(identity), intercept = F,
                    lambda = 1, no.pen = 1:ncol(bas),
                    control.l0 = control.l0(maxit = 25),
                    control.iwls = control.iwls(thresh = 1, maxit = 1),
                    control.fit = control.fit(maxit = 25, block.size = 100, constr = "nonneg"))

xlims <- c(4600, 5100)
plot(x = 1:n, y = y, xlim = xlims, type = "l", log = "y", ylim = c(1, max(y)))
lines(x = 1:n, y = deconv.fit$fitted.values, col = "orange2")
matlines(x = 1:n, y = sweep(X.full[,xlims[1]:xlims[2]], 2, deconv.fit$coefficients[xlims[1]:xlims[2]], "*"), lty = 1, col = "green4")


####################
#### DEBUG ZONE ####
####################





unlist(L0fit$lambda)[unlist(L0fit$suppSize)==0][1] # = 0.0618124
unlist(L0fit$lambda)[unlist(L0fit$suppSize)==max(unlist(L0fit$suppSize))][1] # = 6.5916e-09





max(unlist(L0fit$suppSize)) # size of largest model = 100
max(diag(1/crossprod(X, X)) * (crossprod(X, y)^2)) # 677.1252
