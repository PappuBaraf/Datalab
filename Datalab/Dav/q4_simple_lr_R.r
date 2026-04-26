# DAV_Practical/q4_simple_lr_R.R
# Question 4: Implement and visualize simple linear regression in R. [cite: 2]

# Using built-in 'cars' dataset
data <- cars

# Model Implementation
model <- lm(dist ~ speed, data = data)

# Output Summary
cat("--- Simple Linear Regression Summary ---\n")
print(summary(model))

# Visualization
plot(data$speed, data$dist, main = "Simple Linear Regression",
     xlab = "Speed", ylab = "Stopping Distance", pch = 16, col = "blue")
abline(model, col = "red", lwd = 2)