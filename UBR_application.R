#Setting the working directory
setwd("C:/Users/hegahmed/Google Drive/backup 12-12-19/flash memory/E_/medium/Ensembles_LIDTA2017-master/Ensembles_LIDTA2017-master")

#Loading the data
load("All22DataSets.RData")
#Libraries
library(xgboost)
library(UBL)
library(uba)
library(performanceEstimation)
library(caTools)
library(xgboost)
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

#Function for calculating the mean squared error and getting the predicted data based on XGBOOST
mse <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(mean((labels-preds)^2))
  return(list(metric = "mse", value = err))
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

}
#workflow to check the model results based on different parameters to tune them later based on the UBR
exp <- performanceEstimation(PredTask(DSs[[4]]@formula,DSs[[4]]@data),(workflowVariants("mc.xgboost",eta=c(0.01,0.05,0.1), max_depth=c(5,10,15), nround=c(25,50,100,200,500), cst=c(seq(0.2,0.9,by=0.1)))),
                             EstimationTask("totTime",method=CV(nReps = 1, nFolds=5))
)
#combining the results from the work follow and get the min MSE and MAX F1 score
res <- c()
for(wf in 1:360) {
  res.aux <- c()
  for(i in 1:5) {
    res <- rbind(res,c(getIterationsInfo(exp,workflow=wf,task=1,it=i)$evaluation,wf=wf))
  }
  
}
res <- as.data.frame(res)
res["Model"] <- (rep("xgboost",360*1))
res.aux <- res
res <- aggregate(res,by=list(res$Model,res$wf),FUN=mean)
res$Model <- NULL; res$wf <- NULL; colnames(res)[c(1,2)] <- c("Model","wf")
xgb.mse <- res[res$Model=="xgboost",][which(res[res$Model=="xgboost",]$mse == min(res[res$Model=="xgboost",]$mse)),][1,]
xgb.f1 <- res[res$Model=="xgboost",][which(res[res$Model=="xgboost",]$F1 == max(res[res$Model=="xgboost",]$F1)),][1,]

evaluation <- rbind(xgb.mse,xgb.f1)

#getting the results corresponding to the best evalution for both MSE and F1, then plot to check the performance 
getWorkflow("mc.xgboost.v3",exp)

getWorkflow("mc.xgboost.v18",exp)

all_data=DSs[[4]]@data
set.seed(123)
sample=sample.split(all_data,SplitRatio = 0.7)
train=subset(all_data,sample==TRUE)
test=subset(all_data,sample==FALSE)
train_m=train[,-1]
train_m=data.matrix(train_m)
model1=xgboost(data=train_m, label=train[,1],objective="reg:linear",eta=0.1,colsample_bytree=0.2,max_depth=5,nrounds=25)
pred_mse= predict(model1, xgb.DMatrix(data.matrix(test[,-1]), label=test[,1]))
model2=xgboost(data=train_m, label=train[,1],objective="reg:linear",eta=0.1,colsample_bytree=0.2,max_depth=15,nrounds=50)
pred_f1= predict(model2, xgb.DMatrix(data.matrix(test[,-1]), label=test[,1]))

