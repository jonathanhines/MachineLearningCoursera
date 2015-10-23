## Steps:
## 1 - Load the data:
training <- read.csv("./pml-training.csv")
testing <- read.csv("./pml-testing.csv")

## 2 - Filter out unnescecary columns
## some columns are all NA's or empty, others are just not needed as they relate
## to the way the data was collected, not the data to be fit
removed.columns <- c(
  "X",
  "raw_timestamp_part_1",
  "raw_timestamp_part_2",
  "cvtd_timestamp",
  "new_window",
  "num_window",
  "classe",
  "problem_id"
)
training.filtered <- training[,colSums(is.na(testing))==0 & !( names(training) %in% removed.columns )]
training.classe <- training$classe

testing.filtered <- testing[,colSums(is.na(testing))==0 & !( names(testing) %in% removed.columns )]
testing.problem_id <- testing$problem_id

## 3 - Do some visualizations of the data to get a feel for it
# ggplot(dat=training.filtered,
# aes(x=classe,y=magnet_dumbbell_x,colour=user_name)) +
# geom_dotplot(binaxis="y"
#   ,binwidth=diff(range(training.filtered$magnet_dumbbell_x))/500,
#   ,stackdir="center")


## 4 - Split our training set to get a good feel for what models will be most effective
library(caret)
set.seed(2132)
inTrain = createDataPartition(training.classe, p = 3/4)[[1]]
work.training = training.filtered[ inTrain,]
work.testing = training.filtered[-inTrain,]
work.classe.training <- training.classe[inTrain]
work.classe.testing <- training.classe[-inTrain]

## 5 - Decompose our data into Primary components
## Get the preprocess object
preProc <- preProcess(work.training, method = "pca", thres = 0.9)
## Get PC training data
work.training.PC <- predict(preProc,work.training)
work.testing.PC <- predict(preProc,work.testing)
final.testing.PC <- predict(preProc,testing.filtered)

qplot(work.training.PC$PC1,work.training.PC$PC2,color=work.classe.training)
qplot(work.training.PC$PC1,work.training.PC$PC2,color=work.training.PC$user_name)

## 6 - Run some sample models and look at their confusion matricies
#modFit <- train(work.classe.training ~ .,method="rpart",data=work.training.PC)
modFit <- train(work.classe.training ~ .,method="rf",data=work.training.PC)
modFit
confusionMatrix(work.classe.training,predict(modFit,work.training.PC))
confusionMatrix(work.classe.testing,predict(modFit,work.testing.PC))

test.results <- predict(modFit,final.testing.PC)
## All you need to do is the following:
##   
##   1. Identify which variables to use. (EDA will go a long way. Use negative logic to weed out unwanted variables like empty, irrelevant ones)
##   2. Once you do this you will be surprised to get rid of more than half the variables.
##   3. Then use train() to try different models. At this time you do not know which model can perform better. 
##   4. Compare the models to see which is more accurate and better quality. (hint confusionMatrix)
##   5. Once you narrow down to the one model you want to use, start using that model function directly, as train is very(very) slow and will not allow lot of tuning. 
##   6. Identify the variables that are important and try refitting the model with reduced variables. 
##   7. Also try to tune other parameters nodes, mtry, proximity,,.. 
## NOTE: Do not go too overboard as you will end up over-fitting the model. 

varImpPlot(modFit$finalModel)
modFit.varImp <- varImp(modFit$finalModel)
# Display the importance as text
modFit.varImp[order(modFit.varImp$Overall,decreasing = T),,drop=F]
# Note that the user name values drops off much more quickly than the rest so drop it

modFit_noUser <- train(work.classe.training ~ . - user_name,method="rf",data=work.training.PC)
modFit_noUser
confusionMatrix(work.classe.training,predict(modFit_noUser,work.training.PC))
confusionMatrix(work.classe.testing,predict(modFit_noUser,work.testing.PC))
varImpPlot(modFit_noUser$finalModel)
modFit_noUser.varImp <- varImp(modFit_noUser$finalModel)
modFit_noUser.varImp[order(modFit_noUser.varImp$Overall,decreasing = T),,drop=F]

test.results.no_user <- predict(modFit_noUser,final.testing.PC)
print(test.results.no_user)



pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./results/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test.results.no_user)