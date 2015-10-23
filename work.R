training <- read.csv("./pml-training.csv")
testing <- read.csv("./pml-testing.csv")
training.filtered <- training[,colSums(is.na(testing))==0]
training.filtered$classe <- training$classe
testing.filtered <- testing[,colSums(is.na(testing))==0]
testing.filtered$problem_id <- testing$problem_id