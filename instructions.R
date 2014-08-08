# Store breast cancer sample data into a data frame

wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)

# Remove ID numbers so that they won't be used to "predict" each example
# This prevents overfitting

wbcd <- wbcd[-1]

# The diagnosis variable contains the outcome we hope to predict
# (whether the mass is benign or malignant)

# Many R machine learning classifiers require that the target feature
# is coded as a factor. Let's recode the diagnosis variable

wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

# Exploring the values of three of the features shows that scales differ
# enough to potentially cause problems, esp. between smoothness_mean
# [0.05 - 0.16] and area_mean [143.5 - 2501.0].  Let's apply normalization
# to rescale to a standard value range.

# In order to do this, we have to create a normalize() function that takes a
# vector x of numeric values - for each value in x, subtract the minimum value
# in x and divide by the range of values in x.

normalize <- function(x) { return ((x-min(x)) / (max(x) - min(x))) }

# This will create a corresponding range of values between 0 and 1
# To apply this function to the numeric features in our data frame, let's
# take the lapply() function, which applies a function to each element of
# a list, and use it to apply normalize() to each feature in the data frame.
# Then, we'll convert the list returned by lapply() into a data frame using
# the as.data.frame() function.

wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))

# Run summary(wbcd_n$area_mean) to see the normalized range

# Now we have to prepare the data, creating training and test datasets
# We have to divide our dataset into two portions: a training dataset to build
# the kNN model and a test dataset to estimate the predictive accuracy of the 
# model. Let's use the first 469 records for the training dataset and the
# remaining 100 to simulate new patients.

wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

# Our consecutive selections are okay in this case because the data was 
# already randomly ordered. But this might not be the case for every dataset!
# If the data is not already randomized, random sampling methods might need
# to be used.

# Let's take the diagnosis factor in column 1 of the non-normalized 
# dataset, and let's store it in factor vectors. This will be useful later

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

# (We want to store the test labels so that we can evaluate model performance 
# later)

# Since kNN is a lazy learner, we don't need to build that much of a model - 
# we just have to store the data in a structured format.

# Let's install a package which provides basic R functions for classification

install.packages("class")
library("class")

# The class package has a knn() function that will identify the k-nearest 
# neighbors using Euclidean distance for each instance in the test data.
# The test instance is classified by taking a "vote" among the k-Nearest 
# Neighbors - assigning the class of the majority of the k neighbors. Tie
# votes are broken at random.

# We have our training data, test data and labels for the training data. Now
# we just need to come up with a k-value.

# Let's try 21, an odd number roughly equal to the square root of 469. An
# odd number cuts down on chances of ending with a tie vote.

wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k=21)

# Now we have to evaluate the performance of the knn model. We'll do this
# by comparing how the predicted classes in the wbcd_test_pred vector match
# up with the known values in the wbcd_test_labels vector. We can use the
# CrossTable() function from the gmodels package to do this.

install.packages("gmodels")
library("gmodels")

# The upper-left square shows how many values were "true negatives", 
# the upper-right square - "false positives", 
# lower-left square - "false negatives", lower-right square - "true positives"
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE)




