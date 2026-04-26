# Simple Linear Regression (Student Dataset)

install.packages("caTools")
library(caTools)

# Load dataset (FIXED PATH)
data <- read.csv("C:/Users/LENOVO/Downloads/R/student_data.csv")

# Display structure
str(data)

# Summary
summary(data)
colSums(is.na(data))

# Remove missing values
data <- na.omit(data)

# Check column names
colnames(data)

# Split dataset
set.seed(123)
split <- sample.split(data$G3, SplitRatio = 0.7)

train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Build Simple Linear Regression Model
model <- lm(G3 ~ studytime, data = train_data)

# Model Summary
summary(model)

# Plot Regression Line
plot(train_data$studytime, train_data$G3,
     main = "Simple Linear Regression",
     xlab = "Study Time",
     ylab = "Final Grade (G3)",
     col = "blue")

abline(model, col = "red", lwd = 2)

# Prediction
predictions <- predict(model, newdata = test_data)

# Output
predictions