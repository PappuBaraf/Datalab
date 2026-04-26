# DAV_Practical/q2_eda_R.R
# Question 2: Getting introduced data analytics libraries in R. [cite: 2]

# 1. Data Import (Using built-in mtcars dataset)
data <- mtcars
cat("--- First 5 rows of data ---\n")
print(head(data, 5))

# 2. Data Cleaning (Introducing artificial NAs and cleaning)
data[1, "mpg"] <- NA
data$mpg[is.na(data$mpg)] <- mean(data$mpg, na.rm = TRUE)

# 3. Summary Statistics
cat("\n--- Summary Statistics ---\n")
print(summary(data))

# 4. Visualization
hist(data$mpg, main="Distribution of MPG", xlab="Miles Per Gallon", col="lightblue", border="black")

# 5. Splitting Data (80% Train, 20% Test)
set.seed(123)
sample_size <- floor(0.8 * nrow(data))
train_indices <- sample(seq_len(nrow(data)), size = sample_size)

train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

cat("\n--- Data Splitting ---\n")
cat("Training Samples:", nrow(train_data), "\n")
cat("Testing Samples:", nrow(test_data), "\n")