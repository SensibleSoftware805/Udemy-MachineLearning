# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting data set into test and training set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Simple Linear Regression to the data set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predicting the test set result
y_pred = predict(regressor, newdata = test_set)