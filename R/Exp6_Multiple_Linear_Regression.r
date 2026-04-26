# Multiple Linear Regression (Student Dataset)

install.packages("caTools")
library(caTools)

# Load dataset (correct path)
data <- read.csv("C:/Users/LENOVO/Downloads/R/student_data.csv")

# Structure
str(data)

# Remove missing values
data <- na.omit(data)

# Check columns
colnames(data)

set.seed(123)

# Split using dependent variable G3
split <- sample.split(data$G3, SplitRatio = 0.7)

train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Build Multiple Linear Regression Model
model <- lm(G3 ~ studytime + G1 + G2, data = train_data)

# Summary of model
summary(model)

# Prediction
predictions <- predict(model, newdata = test_data)

# View predictions
predictions

# Residual Plot
plot(predictions, test_data$G3 - predictions,
     col = "purple", pch = 16,
     main = "Residual Plot",
     xlab = "Predicted Values",
     ylab = "Residuals")

abline(h = 0, col = "red")