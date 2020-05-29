####################################################
#
# R Code for the paper "Evaluation of Ensemble Methods in Imbalanced Domain Regression Tasks
#
####################################################


#Setting the working directory
setwd("C:/Users/hegahmed/Google Drive/backup 12-12-19/flash memory/E_/medium/Ensembles_LIDTA2017-master/Ensembles_LIDTA2017-master")

#Loading the data
load("All22DataSets.RData")

#Libraries
library(xgboost)
library(UBL)
library(uba)
library(performanceEstimation)

#Function with evaluation metrics
eval.stats <- function(trues,preds,ph,ls) {
  
  prec <- util(preds,trues,ph,ls,util.control(umetric="P",event.thr=0.9))
  rec  <- util(preds,trues,ph,ls,util.control(umetric="R",event.thr=0.9))
  F1   <- util(preds,trues,ph,ls,util.control(umetric="Fm",beta=1,event.thr=0.9))
  
  mad=mean(abs(trues-preds))
  mse=mean((trues-preds)^2)
  mape= mean((abs(trues-preds)/trues))*100
  rmse= sqrt(mean((trues-preds)^2))
  mae_phi= mean(phi(trues,phi.parms=ph)*(abs(trues-preds)))
  mape_phi= mean(phi(trues,phi.parms=ph)*(abs(trues-preds)/trues))*100
  mse_phi= mean(phi(trues,phi.parms=ph)*(trues-preds)^2)
  rmse_phi= sqrt(mean(phi(trues,phi.parms=ph)*(trues-preds)^2))
  prec=prec
  rec=rec
  F1=F1
  
  c(
    mse=mse, rmse=rmse, prec=prec,rec=rec,F1=F1
  )
  
}

#Function for calculating the mean squared error for XGBOOST
mse <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(mean((labels-preds)^2))
  return(list(metric = "mse", value = err))
}

####################################################
#
# Workflows
#
####################################################

mc.rpart <- function(form,train,test,minsplit,cp,...) {
  
  require(rpart)
  
  tgt <- which(colnames(train)==as.character(form[[2]]))
  
  ph <- phi.control(train[,tgt], method="extremes")
  ls <- loss.control(train[,tgt])
  
  m <- rpart(form,train,control=rpart.control(minsplit=minsplit, cp=cp))
  p <- predict(m, test)
  
  eval <- eval.stats(test[,tgt],p,ph,ls)
  res <- list(evaluation=eval)
  res
  
}

mc.bagging <- function(form,train,test,minsplit,cp,nbags,...) {
  
  require(ipred)
  
  tgt <- which(colnames(train)==as.character(form[[2]]))
  
  ph <- phi.control(train[,tgt], method="extremes")
  ls <- loss.control(train[,tgt])
  
  m <- bagging(form, train, coob=TRUE, nbagg=nbags, control=rpart.control(minsplit=minsplit,cp=cp))
  p <- predict(m, test)
  
  eval <- eval.stats(test[,tgt],p,ph,ls)
  res <- list(evaluation=eval)
  res
  
}

mc.xgboost <- function(form,train,test,eta,cst,max_depth,nround,...) {
  
  require(xgboost)
  
  tgt <- which(colnames(train)==as.character(form[[2]]))
  
  train[,tgt] <- as.numeric(train[,tgt])
  test[,tgt] <- as.numeric(test[,tgt])
  
  ph <- phi.control(train[,tgt], method="extremes")
  ls <- loss.control(train[,tgt])
  
  m <- xgboost(data=xgb.DMatrix(data.matrix(train[,-tgt]), label=train[,tgt]),objective="reg:linear",eta=eta,feval=mse,colsample_bytree=cst,max_depth=max_depth,nthread=5,nrounds=nround,silent=1)
  p <- predict(m, xgb.DMatrix(data.matrix(test[,-tgt]), label=test[,tgt]))
  
  eval <- eval.stats(test[,tgt],p,ph,ls)
  res <- list(evaluation=eval)
  res
  
}

####################################################

d <- 1
exp <- performanceEstimation(PredTask(DSs[[d]]@formula,DSs[[d]]@data),(workflowVariants("mc.xgboost",eta=c(0.01,0.05,0.1), max_depth=c(5,10,15), nround=c(25,50,100,200,500), cst=c(seq(0.2,0.9,by=0.1)))),
                             EstimationTask("totTime",method=CV(nReps = 2, nFolds=5))
)

#
# Running experiments
#
####################################################
res <- c()
for(wf in 1:360) {
  res.aux <- c()
  for(i in 1:10) {
    res <- rbind(res,c(getIterationsInfo(exp,workflow=wf,task=1,it=i)$evaluation,wf=wf))
  }
  
}
res <- as.data.frame(res)
res["Model"] <- (rep("xgboost",360*10))
res.aux <- res
res <- aggregate(res,by=list(res$Model,res$wf),FUN=mean)
res$Model <- NULL; res$wf <- NULL; colnames(res)[c(1,2)] <- c("Model","wf")
xgb.mse <- res[res$Model=="xgboost",][which(res[res$Model=="xgboost",]$mse == min(res[res$Model=="xgboost",]$mse)),][1,]
xgb.f1 <- res[res$Model=="xgboost",][which(res[res$Model=="xgboost",]$F1 == max(res[res$Model=="xgboost",]$F1)),][1,]

final.res <- rbind(xgb.mse,xgb.f1)
