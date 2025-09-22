# Assignment: Data Analysis and Visualization with Pandas & Matplotlib

# ------------------------
# Importing Required Libraries
# ------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ------------------------
# Task 1: Load and Explore the Dataset
# ------------------------

try:
    # Load Iris dataset from sklearn
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame  # Convert to pandas DataFrame
    
    # Display first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")
    
    # Check data types and structure
    print("Dataset Info:")
    print(df.info(), "\n")
    
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum(), "\n")
    
    # Clean dataset (if missing values exist, fill with mean as example)
    df = df.fillna(df.mean(numeric_only=True))

except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# ------------------------
# Task 2: Basic Data Analysis
# ------------------------

# Basic statistics
print("Statistical Summary of Dataset:")
print(df.describe(), "\n")

# Grouping example (mean petal length per species)
grouped = df.groupby("target")["petal length (cm)"].mean()
print("Average Petal Length per Species:")
print(grouped, "\n")

# Identify patterns (simple observation example)
print("Observation: Iris-setosa generally has shorter petals compared to other species.\n")

# ------------------------
# Task 3: Data Visualization
# ------------------------

# Set style
sns.set(style="whitegrid")

# 1. Line Chart (example: sepal length trend by index)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8, 5))
grouped.plot(kind="bar", color="green")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8, 5))
plt.hist(df["sepal width (cm)"], bins=20, color="purple", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(8, 5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], c=df["target"], cmap="viridis")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()
