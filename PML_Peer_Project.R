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
train <- read.table("data/training.csv", header = TRUE, sep = ",", na.strings = c("NA", ""))
dim(train)
names(train)[1:10]
names(train)[155:160]
str(train[160])

##  Details on data collection and feature extraction and selection can be found here:
# http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf
# In short, four sensors:
#   
#   1. Belt
#   2. Glove
#   3. Arm band
#   4. Dumbbell
# 
# were used to collect 3-axis acceleration, gyroscope and magnetometer data.
# 
# Six participants performed one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fash- ions: exactly according to the specification (Class A), throw- ing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).
# 
# For feature extraction we used a sliding window approach with different lengths from 0.5 second to 2.5 seconds, with 0.5 second overlap. In each step of the sliding window ap- proach we calculated features on the Euler angles (roll, pitch and yaw), as well as the raw accelerometer, gyroscope and magnetometer readings.
# For the Euler angles of each of the four sensors we calculated eight features: mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness, generating in total 96 derived feature sets.

# Observed Features:
# Sensors: 4
# Euler angles: 3
# Raw (mag, gyro, accel): 3
(4*3)+4*3*3  # magnitude for each angle and sensor, Raw for each angle and sensor
# 48

# Derrived Features:
# Euler angles: 3
# Sensors: 4
# Derived features: 8 
3*4*8 
# 96

# Additional Derrived Features
# total acceleration
# var total acceleration
2*4

96 + 48 + 8
# 152

# Additional Variables: 
#   X: observation index
#   user_name: user name (one of 6)
#   raw_timestamp_part_1: POSIXct date object
#   raw_timestamp_part_2: ??
#   cvtd_timestamp: converted timestamp
#   new_window
#   num_window
#   classe
# Total: 8

152 + 8
# 160

# Note: skewness_roll_belt.1 should be skewness_pitch_belt

# how many non NA obs for skewness_roll_belt.1?
nrow(train) - sum(is.na(train$skewness_roll_belt.1))
# 374


## look for NAs: first done with only "NA" and "" in the na.strings.
colSums(is.na(train))
# many variables hove 19216 NAs. All others have none.
# what is unique about those rows (observations) that have non NA values for those variables
# with 19216?

notNA <- train[train$kurtosis_roll_belt != "NA", ]
summary(notNA$new_window)
# no  yes 
# 0   406 
# they are all new windows and they are all of the new window observations
summary(train$new_window)
# no      yes 
# 19216   406 

# all users have new window observations. on average about 11-16 per set--or about 1 - 1.5 per rep.
summary(notNA$user_name)
# adelmo carlitos  charles   eurico   jeremy    pedro 
# 83       56       81       54       77       55 

# How many columns have #DIV/0! ?

# isDiv0 <- function(x) {
#   if( grepl("#DIV/0!", x)) { return(1) } else { return(0) }
# }

isDiv0 <- function(v) { sum(grepl("#DIV/0!", v)) }
div0l <- sapply (train, function(x) isDiv0(x) )
div0 <- as.data.frame(div0l)
div0$names <- names(div0l)
div0[div0$div0l > 0, ]
# there are lots of "#DIV/0!"s these are effectively NA. Add #DIV/0! to the na.string and re-read
# the data in to R.
nastrings <- c( "NA", "", "#DIV/0!" )
train <- read.table("data/training.csv", header = TRUE, sep = ",", na.strings = nastrings)

# look for factors--everything should be numeric or integer.
str(train[1:80])
str(train[81:160])

# What are the new_window "yes" observations?
win13 <- train[train$num_window == "13", ]
tail(win13, n=2)

# the new_window "yes" observations have summary stats: min, max, amplitude, variance, avg, stddev,
# skewness, kurtosis for the window.


# how many obesrvations of each classe are there per user?

table(train$user_name, train$classe)

#             A    B    C    D    E
# adelmo   1165  776  750  515  686
# carlitos  834  690  493  486  609
# charles   899  745  539  642  711
# eurico    865  592  489  582  542
# jeremy   1177  489  652  522  562
# pedro     640  505  499  469  497


######
######  Possible diversion. waht is raw_timestamp_part2? Is this a millisecond reading?
######  
    #  look at one set of 10 reps:
carlitosA <- train[ (train"user_name" == "carlitos" & train$classe == "A"),]
nrow(carlitosA)
    # 834 observations over those 10 reps.

    # How many windows?
cwin <- as.factor(carlitosA[,7])
length(levels(cwin))
    # 34 windows over 10 reps

    # observations per window
table(table(cwin))
    # are these independent? i.e. are these resampling of the same activity?
carlitosA[carlitosA$num_window == "322", 1:46]

window <- win13$raw_timestamp_part_2
( max( window) - min( window))/1000000

# within a window the timestamp_part_2 increases monotonically and given that there are 34 
# windows over 10 reps, they must be shorter than 1 sec (as the authors say they are), so 
# take max - min per window and divide by 1M to get duration in seconds. (that is a guess)
# or, perhaps the new_window part 2 is the total time of the window
newwins <- train[train$new_window == "yes", "raw_timestamp_part_2"]
min(newwins) # 492326
max(newwins) # 998801
# if strictly 10e-6 this would be only .5 and 1.0 seconds. but there are 2.5 second windows.
# perhaps look at the part 1 and see that seconds elapse so need to add to part 1. 
# need to divide by 1M and add to part 1 and then take just the seconds and change.
z <- max(win13$raw_timestamp_part_1)
as.POSIXct(z, origin = "1970-01-01") 
y <- min(win13$raw_timestamp_part_1)
as.POSIXct(y, origin = "1970-01-01")
w <- max(win13$raw_timestamp_part_2)
x <- as.numeric(paste(z, w, sep = ""))
as.POSIXct(x/1000000, origin = "1970-01-01")


# next visualize the data, do some plots. See plotting predictors video from PLM 

# , "total_accel_belt", 
# "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", 
# "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", 
# "magnet_belt_z"

featurePlot(x=train[,c("roll_belt","pitch_belt","yaw_belt")],
            y = train$classe,
            plot="pairs")

featurePlot(x=train[,c("roll_belt","pitch_belt","yaw_belt")],
            y = train$classe,
            plot="density")

qplot(roll_belt, color = classe, data = train, geom = c("density"))
qplot(roll_belt, classe, data = train)

qplot(pitch_belt, color = classe, data = train, geom = c("density"))
qplot(pitch_belt, classe, data = train)

qplot(yaw_belt, color = classe, data = train, geom = c("density"))
qplot(yaw_belt, classe, data = train)

qplot(roll_arm, color = classe, data = train, geom = c("density"))
qplot(roll_arm, classe, data = train)

qplot(pitch_arm, color = classe, data = train, geom = c("density"))
qplot(pitch_belt, classe, data = train)

qplot(yaw_arm, color = classe, data = train, geom = c("density"))
qplot(yaw_arm, classe, data = train)

# distributions are not normal. Not sure regression will work. look at residuals?

# different classes peak in different peaks
qplot(magnet_arm_x, color = classe, data = train, geom = c("density"))
qplot(magnet_arm_x, classe, data = train)

qplot(magnet_arm_y, color = classe, data = train, geom = c("density"))
qplot(magnet_arm_y, classe, data = train)

qplot(magnet_arm_z, color = classe, data = train, geom = c("density"))
qplot(magnet_arm_z, classe, data = train, geom = c("jitter"))

qplot(magnet_forearm_x, color = classe, data = train, geom = c("density"))
qplot(magnet_forearm_x, classe, data = train, geom = c("jitter"))

qplot(gyros_forearm_x, color = classe, data = train, geom = c("density"))
qplot(var_accel_forearm, color = classe, data = train, geom = c("density"))

qplot(total_accel_forearm, color = classe, data = train, geom = c("density"))
qplot(magnet_dumbbell_x, color = classe, data = train, geom = c("density"))

qplot(pitch_belt, yaw_belt, color = classe, data = train)
qplot(pitch_belt, color = classe, data = train, geom = c("density"))
qplot(yaw_belt, color = classe, data = train, geom = c("density"))

# try just the new_window "yes" rows only and test prediction after using all rows.
# look at clustering video from exploratory data.

# what preprocessing needs to be done? 
# need to impute NAs. There are a lot of them. remove any variable with all NAs.
# center and scale? from above most are not normally distributed.
# nearZeroVar-- start here. Are there any that are near zero variance?

trainOV <- nearZeroVar(train, saveMetrics = TRUE)
nzvT <- nearZeroVar(train)
trainOV$totalNAs <- colSums(is.na(train))
head(trainOV, n = 30)
trainOV$Class <- sapply(train, class)

#####
##### remove nzv variables
#####

train_nzvR <- train [, -nzvT]

#### impute missing values
preObj <- preProcess(train_nzvR[,-124],method="knnImpute")

#### indentify just numeric or integer variables
tfs <- sapply(train_nzvR, is.factor)
tFactorVars <- which(tfs)

####  impute missing values -- note this centers and scales all variables including the num_window
####  and time stamps.
preObj <- preProcess(train_nzvR[,-tFactorVars],method="knnImpute")
trainZI <- predict(preObj, train_nzvR[,-tFactorVars])

### look for correlated variables
trainCorr <- cor(trainZI)
trainHC <- findCorrelation(trainCorr, cutoff = 0.75)

### remove the high correlation variables
trainZInHC <- trainZI[, -trainHC]

### add the classe outcome variable back to the processed data (although this is not necessary)
### if you do predict(train$class ~., method = "xxx", data = trainZInH)
trainZInHC_c <- trainZInHC
trainZInHC_c$classe <- train[, "classe"]

### what does our data look like now?

qplot(pitch_belt, color = classe, data = trainZInHC, geom = c("density"))
qplot(pitch_belt, classe, data = trainZInHC)

qplot(pitch_forearm, color = classe, data = trainZInHC, geom = c("density"))
qplot(pitch_forearm, classe, data = trainZInHC)

qplot(magnet_forearm_x, color = classe, data = trainZInHC, geom = c("density"))
qplot(magnet_forearm_x, classe, data = trainZInHC, geom = c("jitter"))

qplot(magnet_belt_z, color = classe, data = trainZInHC, geom = c("density"))

# These plots are unchange from their counterparts in the train data.
qplot(pitch_belt, yaw_belt, color = classe, data = trainZInHC_c)
qplot(pitch_belt, color = classe, data = trainZInHC_c, geom = c("density"))
qplot(yaw_belt, color = classe, data = trainZInHC_c, geom = c("density"))

# How to figure out what covariants to use? and given the "scattered" islands for each class in 
# pitch ~ yaw plot, what does that mean? How to interpret?
# I suppose these graphs are showing that the data is non - linear. thus we need to go to trees 
# or other non-linear prediction methods.

set.seed(1771)
modFitRF <- train(classe ~., data = trainZInHC_c, method = "rf")

# very slow! but gets the answer. Likely overfitted. was there cross validation? 
# Try boosting and then limiting the number of predictors. Try rpart too. Try just the derrived 
# features vs the raw features.



# helpful once predictions are made
pred <- predict(modFit,testing)
testing$predRight <- pred==testing$Species
table(pred,testing$Species)



# for continuous variables with NAs create a column isNA (0, 1) factor (1 is a NA). then fill in 
# the NA in the data column with the median of the data values. Use both to predict and the 0,1 
# will catch anything intersting in the NA.
# for factors, if NAs, add an NA level.
# if too many factors then merge low volume low predictive factors into one super factor.
# https://www.youtube.com/watch?v=kwt6XEh7U3g around the 45 min mark
# 1:05 mark. if there are lots of zeros -- call that variable zero. can result in better results
# 1:09 mark. sampling without replacement can be more effective. Select less than R's default
# 63.1% of the total data in subspaces. JN uses 10% or so.

# data(attitude)
library(ggplot2)
library(reshape2)
# qplot(x=Var1, y=Var2, data=melt(cor(attitude)), fill=value, geom="tile")

logicals <- grepl("logical", trainOV$Class)
t_c_only <- train[ , - which(logicals)]
t_c_only <- t_c_only[, c(1, 4, 7:153)]
c_dummies <- dummyVars( ~ classe, data = train)
train_w_dummies <- predict(c_dummies, newdata = train)
t_c_only <- cbind(t_c_only, train_w_dummies)
tco_classes <- as.data.frame(sapply(t_c_only, class))
qplot(x=Var1, y=Var2, data=melt(cor(t_c_only)), fill=value, geom="tile")

