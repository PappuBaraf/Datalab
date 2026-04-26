# Set working directory (optional, but good practice)
setwd("C:/Users/LENOVO/Downloads/R")

# Load Data - added comment.char = "#" to fix the error!
data <- read.csv("C:/Users/LENOVO/Downloads/R/DAV_dataset.csv", comment.char = "#") 

# Cleaning (Remove NA values)
data <- na.omit(data) 
print("--- Summary of Cleaned Data ---")
print(summary(data))

# Visualization (Plotting Age or Salary)
# Using data[,2] assumes Age is the second column. 
hist(as.numeric(data[,2]), col="blue", main="Histogram of Age", xlab="Age") 

# Split Data (80% Train, 20% Test)
set.seed(123) 
index <- sample(1:nrow(data), 0.8 * nrow(data)) 
train <- data[index,] 
test <- data[-index,]

print("--- Data Splitting Results ---")
print(paste("Training data rows:", nrow(train)))
print(paste("Testing data rows:", nrow(test)))
