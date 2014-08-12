### PLM_Peer_Project2.R
### Practical Machine Learning Peer Project
### Coursera, Data Scientist Specialization 
### August 2014

# steps in our prediction process:
#   question:   The goal of your project is to predict the manner in which they did the exercise.
#               (This is the "classe" variable in the training set.)
#   input data: http://groupware.les.inf.puc-rio.br/har 
#               Training Data:  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#               Test Data:      https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
#   features:   You may use any of the variables (other than Classe) to predict with.
#   algorithm:
#   parameters:
#   evaluation:


#### <Make this a heading> Question:
##  Background
## 
##  Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
## 
##  The goal of your project is to predict the manner in which they did the exercise.
##  put answer here: model, accuracy, modelfit coefs, etc. COMPARE to the Confusion Matrix on websit.
####

require(caret)
require(RANN)
require(ggplot2)
require(reshape2)

####  <Make this a heading> Data:

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 
# The data is already split into training and test sets, Download the data and determine the size

##  get the data set as do not run, do not echo
if(!file.exists("./data")){dir.create("./data")}
TrfileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(TrfileURL, destfile = "./data/training.csv", method = "curl")
TefileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(TefileURL, destfile = "./data/testing.csv", method = "curl")
##

trainInfo <- file.info("data/training.csv")
trainInfo$size # bytes

# read the data into R
nastrings <- c( "NA", "", "#DIV/0!" )
train <- read.table("data/training.csv", header = TRUE, sep = ",", na.strings = nastrings)


# start to visualize the data
trainOV <- nearZeroVar(train, saveMetrics = TRUE)
nzvT <- nearZeroVar(train)
trainOV$totalNAs <- colSums(is.na(train))
trainOV$Class <- sapply(train, class)
head(trainOV, n = 30)

# remove all the variables with no data at all (the show up as logicals)
# get rid of all the factor variables except classe (for the time being)
# plot the pairwise correlations
logicals <- grepl("logical", trainOV$Class)
t_c_only <- train[ , -which(logicals)]
t_c_only <- t_c_only[, c(1, 4, 7:153)]
c_dummies <- dummyVars( ~ classe, data = train)
train_w_dummies <- predict(c_dummies, newdata = train)
t_c_only <- cbind(t_c_only, train_w_dummies)
tco_classes <- as.data.frame(sapply(t_c_only, class))
qplot(x=Var1, y=Var2, data=melt(cor(t_c_only)), fill=value, geom="tile")

# lots of NAs, some correlations
# get rid of the NAs by 1) creating isna variables to show where the NAs are, and 2) then 
# interploating the missing values to get rid of all NAs.

naf <- function( z, y ) {
  if (is.na(y)) { z <- 1 } else { z <- 0 }
}
addDummy <- function (v) {
  if(sum(is.na(v) == 0)) { stop 
    } else {
    vmedian <- median(v, na.rm = TRUE)
    d <- numeric(length(v))
    d <- sapply(length(v), function(i) {
      if(is.na(v[i])) {
        v[i] <- vmedian
        d[i] <- 1
      } else {
        d[i] <-0
      }
    })
#     print(d)
    dvar <- as.data.frame(d)
#     newName <- paste (v, "_isna", sep = "")
#     names(dvar) <- newName
    
    return(dvar)
  }
}
trainT <- sapply(train, addDummy)
testv2 <- data.frame( v1 = c(1,2,3,4), v2 = c(5,6,NA,7))
