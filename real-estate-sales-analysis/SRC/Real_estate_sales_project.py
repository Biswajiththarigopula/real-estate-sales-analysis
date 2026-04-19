import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

print("------Real_Estate_Sales Analysis------")

#Loading Dataset
df = pd.read_csv("Real_Estate_Sales(2020-2021).csv",low_memory=False)

#Implemating Exploratory Data Analysis(EDA)
print("------Exploratory Data Analysis(EDA)------")

#first 5 rows
print("------First 5 Rows------")
print(df.head())

#last 5 rows
print("------Last 5 Rows------")
print(df.tail())

#dataset shape
print("\n------Shape------")
print(df.shape)

#datasetinfo
print("\n------Info------")
print(df.info())

#summary statistics
print("\n------Summary------")
print(df.describe())

#check missing values
print("\n------Missing------")
print(df.isnull().sum())

#compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)
print("------Correlation Matrix:------\n",correlation_matrix)

#compute covariance matrix
covariance_matrix = df.cov(numeric_only=True)
print("------Covariance Matrix:------\n",covariance_matrix)

#IQR method
print("------IQR Method for Sale Amount------")


#select numerical column
column = "Sale Amount"

#calculate Q1,Q3 and IQR
Q1 = df[column].quantile(0.25)
print(Q1)
Q3 = df[column].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
print(IQR)

#define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("lower:",lower_bound)
print("upper:",upper_bound)

#identifying outliers
outliers = df[(df[column]<lower_bound) | (df[column]>upper_bound)]
print("outliers detected:\n",outliers)

#boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df[column])
plt.title("boxplot for outliers detection")
plt.show()

print("------IQR Method for Assessed Value------")


#select numerical column
column = "Assessed Value"

#calculate Q1,Q3 and IQR
Q1 = df[column].quantile(0.25)
print(Q1)
Q3 = df[column].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
print(IQR)

#define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("lower:",lower_bound)
print("upper:",upper_bound)

#identifying outliers
outliers = df[(df[column]<lower_bound) | (df[column]>upper_bound)]
print("outliers detected:\n",outliers)

#boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df[column])
plt.title("boxplot for outliers detection")
plt.show()

#Visualization
print("------Visualization------")

#histogram
plt.figure()
df_clean = df[df["Sale Amount"] < 1000000]
sns.histplot(df_clean["Sale Amount"], bins=50)
plt.title("Distribution of Sale Amount")
plt.show()

#scatter plot
plt.figure()
sns.scatterplot(x="Assessed Value",y="Sale Amount",hue="Property Type",data=df)
plt.title("Assessed Value vs Sale Amount")
plt.show()

#box plot
plt.figure()
sns.boxplot(x="Property Type",y="Sale Amount",data=df)
plt.title("Sale Amount by Property Type")
plt.show()

#heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Correlation HeatMap")
plt.show()

#kdeplot
plt.figure()
sns.kdeplot(x=df["Sale Amount"],fill=True,color='blue')
plt.show()

#pair plot
sns.pairplot(df,hue="Property Type")
plt.show()

#Linear regression
print("------Linear Regression------")
# Using 'Assessed value' to predict 'sale amount'
X = df[['Assessed Value']]
y = df['Sale Amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² score:", round(r2_score(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))

# Plot regression line
plt.figure(figsize=(8,5))
sns.regplot(x='Assessed Value', y='Sale Amount', data=df, line_kws={"color":"red"})
plt.title("Linear Regression: Sale Amount vs Assessed Value")
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.grid(True)
plt.show()

#z-test
print("------Z-Test for Sale Amount------")

#population_data
population_data = df["Sale Amount"]

#estimate "population" parameters
population_mean = np.mean(population_data)
population_std = np.std(population_data,ddof=0)

#taking a random sample
np.random.seed(42)
sample_size = 30
sample = np.random.choice(population_data,sample_size,replace=False)

#compute sample mean
sample_mean = np.mean(sample)

#perform one-sample z-test
z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))

#compute p-value(two-tailed test)
p_value = 2* (1-stats.norm.cdf(abs(z_score)))

#decision
alpha = 0.05
if p_value < alpha:
    print("reject the null hypothesis: the sample mean is significantly difference.")
else:
    print("fail to reject the null hhypothesis : no significant difference.")

#Z-test
print("------Z-Test for Assessed Value------")

#population_data
population_data = df["Assessed Value"]

#estimate "population" parameters
population_mean = np.mean(population_data)
population_std = np.std(population_data,ddof=0)

#taking a random sample
np.random.seed(42)
sample_size = 30
sample = np.random.choice(population_data,sample_size,replace=False)

#compute sample mean
sample_mean = np.mean(sample)

#perform one-sample z-test
z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))

#compute p-value(two-tailed test)
p_value = 2* (1-stats.norm.cdf(abs(z_score)))

#decision
alpha = 0.05
if p_value > alpha:
    print("reject the null hypothesis: the sample mean is significantly difference.")
else:
    print("fail to reject the null hhypothesis : no significant difference.")


print("------END------")
