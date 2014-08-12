### Practical Machine Learning Peer Project

###
# In this project, your goal will be to use data from accelerometers on the belt, 
# forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts 
# correctly and incorrectly in 5 different ways. 
# More information is available from the website here: 
# http://groupware.les.inf.puc-rio.br/har 
# (see the section on the Weight Lifting Exercise Dataset). 

# Citations: If you use the document you create for this class for any purpose please 
# cite them as they have been very generous in allowing their data to be used for this 
# kind of assignment. 

## Goal of project:
# The goal of your project is to predict the manner in which they did the exercise. 
# This is the "classe" variable in the training set. 
# You may use any of the other variables to predict with. 
# You should create a report describing
#   1.  how you built your model, 
#   2.  how you used cross validation, 
#   3.  what you think the expected out of sample error is, and 
#   4.  why you made the choices you did. 
#   5.  You will also use your prediction model to predict 20 different test cases.

# A.  Your submission should consist of a link to a Github repo with your R markdown and 
#     compiled HTML file describing your analysis:

#     Please constrain the text of the writeup to < 2000 words and the 
#     number of figures to be less than 5. 
#     It will make it easier for the graders if you submit a repo with a gh-pages branch 
#     so the HTML page can be viewed online (and you always want to make it easy on graders :-).

# B.  You should also apply your machine learning algorithm to the 20 test cases 
#     available in the test data above. 
#     Please submit your predictions in appropriate format to the programming assignment 
#     for automated grading. See the programming assignment for additional details. 

### Evaluation criteria

# 1.  Has the student submitted a github repo?

# 2.  Does the submission build a machine learning algorithm to predict activity quality 
#     from activity monitors?
#     To evaluate the HTML file you may have to download the repo and open the compiled HTML 
#     document. 
#     Alternatively if they have submitted a repo with a gh-pages branch, 
#     you may be able to view the HTML page on the web. If the repo is: 
#     https://github.com/DataScienceSpecialization/courses/tree/master/08_PracticalMachineLearning/001predictionMotivation
#     then you can view the HTML page here: 
#     http://datasciencespecialization.github.io/courses/08_PracticalMachineLearning/001predictionMotivation/

# 3.  Do the authors describe what they expect the out of sample error to be and estimate 
#     the error appropriately with cross-validation?

# 4.  Please use the space below to provide constructive feedback to the student who 
#     submitted the work. Point out the submission's strengths and identify some areas 
#     for improvement. You may also use this space to explain your grading decisions.

# 5.  As far as you can determine, does it appear that the work submitted for this project 
#     is the work of the student who submitted it? 



##  get the data
if(!file.exists("./data")){dir.create("./data")}
TrfileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(TrfileURL, destfile = "./data/training.csv", method = "curl")
TefileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(TefileURL, destfile = "./data/testing.csv", method = "curl")

trainInfo <- file.info("data/training.csv")
trainInfo$size
testInfo <- file.info("data/testing.csv")
testInfo$size

# read some of the data to see how it is set up.
train <- read.table("data/training.csv", header = TRUE, nrows = 100, sep = ",")
length(names(traindf))
head(traindf)[1:5, 1:5]
names(traindf)[1:30]
names(traindf)[130:160]
str(traindf)

# int, num, factors, lots of NAs.

testdf <- read.table("data/testing.csv", header = TRUE, sep = ",")
length(names(testdf))
identical(names(traindf), names(testdf))
# FALSE
all.equal(names(traindf), names(testdf))
# "1 string mismatch" last var is 'classe' in traindf and 'problem_id' in testdf
names(testdf)[150:160]
names(traindf)[150:160]

# training <- traindf[ ,1:159]
# testing <- testdf[, 1:159]
# identical(names(training), names(testing))
# TRUE

## read in all the training data.
traindf <- read.table("data/training.csv", header = TRUE, sep = ",")

### how many rows:
nrow(traindf)
# [1] 19622
nrow(testdf)
# [1] 20

ctrain <- sapply(traindf, class)
length(ctrain)
table(ctrain)
# factor integer numeric 
# 37      35      88 

# count NAs by column--particularly the classe column:

sum(is.na(traindf$classe))
# [1] 0

nacount <- function(var) {sum(is.na(var))}
na_by_var <- sapply(traindf, nacount)
na_by_var

head(traindf)[1:5, 1:5]

#### to do's
# find the codebook
# http://simplystatistics.org/2014/06/13/what-i-do-when-i-get-a-new-data-set-as-told-through-tweets/
####

# set up a cross validation set:
library(caret)
set.seed(62912)
inTrain <- createDataPartition(y=traindf$classe, p=0.75, list=FALSE)
mtraining <- traindf[inTrain]
cvtraining <- traindf[-inTrain]
# why is mtraining so much smaller than cvtraining? I would have through that with p=0.75 
#that it would be the other way around. Perhaps it is taking into account the nature of the data?

pairs(mtraining)
# this does not work because the createDataPartition function produces a matrix and then 
# mtraining is character vector. Need to pre-process BEFORE splitting into train and cross validation
# sets.

#### preprocessing
# check for near zero variance variables.

nzv <- nearZeroVar(traindf, saveMetrics = TRUE)
nzv[nzv$nzv, ][1:10, ]    # nzv[nzv$nzv, ] selects the rows where column nzv is TRUE.
class(nzv)
dim(nzv)
head(nzv)
dim(nzv[nzv$nzv,])
# [1] 60  4
# 60 variables have zero or near-zero variance.

# the following uses http://caret.r-forge.r-project.org/preprocess.html
# remove all the nzv variables and work with only the non nzv vars.
nzv <- nearZeroVar(traindf)
filteredTDF <- traindf[, -nzv]
dim(filteredTDF)
descrCor <- cor(filteredTDF)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > 0.999)

# this poops out on the descrCor step:
# Error in cor(filteredTDF) : 'x' must be numeric
# so now must go back and coerce all the filtered vars to numeric.
# First, let's look at them.

names(filteredTDF)
ctrain <- sapply(filteredTDF, class)
length(ctrain)
table(ctrain)
# factor integer numeric 
# 3      35      62 
# Which three are factors?
grep("factor", ctrain)
# [1]   2   5 100
table(filteredTDF[, 100])
# A    B    C    D    E 
# 5580 3797 3422 3216 3607 
table(filteredTDF[, 2])
# adelmo carlitos  charles   eurico   jeremy    pedro 
# 3892     3112     3536     3070     3402     2610 
table(filteredTDF[, 5])
# 02/12/2011 13:32 02/12/2011 13:33 02/12/2011 13:34 02/12/2011 13:35 02/12/2011 14:56 02/12/2011 14:57 
# 177             1321             1375             1019              235             1380 
# 02/12/2011 14:58 02/12/2011 14:59 05/12/2011 11:23 05/12/2011 11:24 05/12/2011 11:25 05/12/2011 14:22 
# 1364              557              190             1497             1425              267 
# 05/12/2011 14:23 05/12/2011 14:24 28/11/2011 14:13 28/11/2011 14:14 28/11/2011 14:15 30/11/2011 17:10 
# 1370              973              833             1498              739              869 
# 30/11/2011 17:11 30/11/2011 17:12 
# 1440             1093 

filteredTDF[, 5] <- as.POSIXct(strptime(as.character(filteredTDF[, 5]), "%d/%m/%Y %H:%M"))
class(filteredTDF[, 5])
filteredTDF[1:5, 5]

dummies <- dummyVars(classe ~., data = filteredTDF)
dfTDF <- predict(dummies, newdata = filteredTDF)
  
dim(dfTDF)
head(dfTDF)[1:10]
ppdf <- as.data.frame(dfTDF)
head(ppdf)[1:10]
dim(ppdf)
class(ppdf$classe)
names(ppdf)[100:104]
ppdf$classe <- filteredTDF$classe

# test for highly correlated variables
descrCor <- cor(ppdf)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > 0.999)
length(highCorr)
highCorr
# NA  there are none

length(descrCor)
dim(descrCor)
class(descrCor)
descrCor[1:10,1:10]

# now try creating testing and cv sets.
set.seed(253)
inTrain <- createDataPartition(y=ppdf$classe, p=0.75, list=FALSE)
mtraining <- ppdf[inTrain]
cvtraining <- ppdf[-inTrain]
# again, not sure why cvtraining is larger than mtraining, but get on with it.

library(randomForest)
modelfit <- train(classe ~., data = ppdf, model = "glm")
pred <- predict(modelfit, cvtraining)
# Error in eval(expr, envir, enclos) : object 'X' not found

# potentially helpful for submission:
# https://help.github.com/articles/creating-project-pages-manually