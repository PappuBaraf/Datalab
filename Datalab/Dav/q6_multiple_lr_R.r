# DAV_Practical/q6_multiple_lr_R.R
# Question 6: Implement and visualize multiple linear regression in R. [cite: 2]

# Using built-in mtcars dataset
# Predicting 'mpg' using 'wt' (weight) and 'hp' (horsepower)
data <- mtcars

# Model Implementation
model <- lm(mpg ~ wt + hp, data = data)

# Output Summary
cat("--- Multiple Linear Regression Summary ---\n")
print(summary(model))

# Visualization: Actual vs Predicted
predicted_mpg <- predict(model)
plot(data$mpg, predicted_mpg, main="Actual vs Predicted MPG",
     xlab="Actual MPG", ylab="Predicted MPG", pch=16, col="darkgreen")
abline(a=0, b=1, col="red", lwd=2) # Line of perfect prediction