""" CRISP-ML(Q) process model describes six phases:
1. Business and Data Understanding
2. Data Preparation
3. Model Building
4. Model Evaluation
5. Deployment
6. Monitoring and Maintenance

Objective: Maximize the operational efficiency
Constraints: Maximize the financial health

Success Criteria: 
Business Success Criteria: Increase the operational efficiency by 10% to 12% by segmenting the Airlines.
ML Success Criteria: Achieve a Silhouette coefficient of at least 0.7
Economic Success Criteria: The airline companies will see an increase in revenues by at least 8% (hypothetical numbers)"""

import pandas as pd  # Importing Pandas library for data manipulation
import numpy as np   # Importing NumPy library for numerical computations
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting
import dtale  # Importing dtale library for automated EDA (Exploratory Data Analysis)from AutoClean import AutoClean  # Importing AutoClean library for automated data cleaning

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer 

from sklearn.cluster import KMeans  # Importing KMeans for clustering
from sklearn import metrics  # Importing metrics for evaluating clustering performance
import joblib  # Importing joblib for saving trained models
import pickle  # Importing pickle for saving Python objects
from scipy.stats import skew

from sqlalchemy import create_engine, text  # Importing create_engine and text from sqlalchemy for database interaction


uni = pd.read_csv(r"F:\assignment\DS ASS QUES\DATA SET\hirerarchial dataset\AirTraffic_Passenger_Statistics.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = 'sudarvignesh'  # password
db = 'air'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
uni.to_sql('passenger_tbl_kmeans', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


###### To read the data from MySQL Database
sql = 'select * from passenger_tbl_kmeans;'
df = pd.read_sql_query(text(sql), engine.connect())

# Data types
df.info()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
df.describe()  # Generating descriptive statistics of the DataFrame 'df', including count, mean, std, min, max, etc.

#Univariate Analysis

df['Passenger Count'].describe()
df['Passenger Count'].hist()


df['Operating Airline'].value_counts()
df["Terminal"].value_counts()
df["Terminal"].value_counts().plot(kind='bar')
df["Operating Airline"].value_counts()


df['Month'].value_counts()
df['Year'].value_counts()

#Bivariate Analysis

df.groupby('Operating Airline')['Passenger Count'].mean()
df.groupby('Terminal')['Passenger Count'].mean().plot(kind='bar')
df.groupby('GEO Region')['Passenger Count'].mean().plot(kind='bar')

# AutoEDA
# D-Tale

# Display the DataFrame using D-Tale
d = dtale.show(df, host = 'localhost', port = 8000)

# Open the browser to view the interactive D-Tale dashboard
d.open_browser()


# Data Preprocessing

# 1. Check missing values
print(df.isnull().sum())

# 2. Drop 'Activity Period' (optional)
df.drop(columns=['Activity Period'], inplace=True)
df.drop(columns=['Operating Airline IATA Code'], inplace=True)
df.drop(columns=['Boarding Area'], inplace=True)
df.drop(columns=['GEO Region'], inplace=True)

 #    0          Symmetric (Normal)                
# 0 to 0.5       Slight positive skew (Acceptable) 
# > 0.5          **Moderate/High positive skew**   
# < 0             Negative skew (left tail)         

from scipy.stats import skew
skewness = skew(df['Passenger Count'])
print("Skewness:", skewness)

# define outlier

# Step 1: Calculate Q1, Q3, and IQR
Q1 = df['Passenger Count'].quantile(0.25)
Q3 = df['Passenger Count'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Count outliers
outlier_count = df[(df['Passenger Count'] < lower_bound) | (df['Passenger Count'] > upper_bound)].shape[0]
print("Number of outliers in Passenger Count:", outlier_count)

# Box plot to visualize
df['Passenger Count'].plot.box()

df['Passenger Count'] = np.where(df['Passenger Count'] > upper_bound, upper_bound,
                          np.where(df['Passenger Count'] < lower_bound, lower_bound,
                                   df['Passenger Count']))

skewness = skew(df['Passenger Count'])
print("Skewness:", skewness)

 #df['Passenger_Count_Log'] = np.log1p(df['Passenger Count'])


 # print("Skew after Winsorization + Log1p:", skew(df['Passenger_Count_Log'])) #If the result is between –0.8 to +0.8, you're good to go for clustering.


from sklearn.pipeline import Pipeline 

numeric_features = df.select_dtypes(exclude = ['object']).columns

categorical_features = df.select_dtypes(include = ['object']).columns

# Defining a Pipeline to deal with missing data and scaling numeric columns
# The Pipeline consists of two steps: imputation using mean strategy and scaling using standard scaler
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('scale', StandardScaler())])


# 1. Encode Categorical: OrdinalEncoder for some columns using pipeline
from sklearn.preprocessing import OrdinalEncoder
categ_pipeline = Pipeline([('ordinal', OrdinalEncoder())])

from sklearn.compose import ColumnTransformer # Importing ColumnTransformer to transfer pipelines into the data
# Using ColumnTransfer to transform the Pipelines into the data. 
# This estimator allows different columns or column subsets of the input to be
# transformed separately and the features generated by each transformer will
# be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

# Pass the raw data through pipeline
processed = preprocess_pipeline.fit(df) 



import os
os.chdir(r"F:\assignment\DS ASS ANS\kmeans")

# ## Save the Imputation and Encoding pipeline
# import joblib
joblib.dump(processed, 'preprocessing')

# Clean and processed data for Clustering
df_clean = pd.DataFrame(processed.transform(df), columns = processed.get_feature_names_out())



#Saving preprocessed data
user = 'root'  # Username
pw = 'sudarvignesh'  # Password
db = 'usl'  # Database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
df_clean.to_sql('df_clean', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Clean data
df_clean.describe()

df_clean.isnull().sum()

# CLUSTERING MODEL BUILDING

# KMeans Clustering 

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#pip install kneed
from kneed import KneeLocator 
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote
import pickle
import os

# Define the range of K values (number of clusters)
k_values = list(range(2, 9))

# Randomly select 5 values of K to evaluate
random_k_values = random.sample(k_values, 5)

# Store results for finding the best K
best_k = None  # To store the best K value
best_score = -1  # Initialize best silhouette score
TWSS = []  # Total Within-Cluster Sum of Squares

# Loop through each randomly selected K
for k in random_k_values:
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    labels = kmeans.fit_predict(df_clean)
    
    # Compute silhouette score (higher is better)
    score = silhouette_score(df_clean, labels) if len(set(labels)) > 1 else -1

    # Store TWSS (inertia) for scree plot
    TWSS.append((k, kmeans.inertia_))

    # Update best K if the score improves
    if score > best_score:
        best_score = score
        best_k = k

# Print the best K value found
print("Best K:", best_k)
print("Best Silhouette Score:", best_score)

# Convert TWSS list to sorted format for plotting
TWSS.sort()
k_values_sorted, inertia_values = zip(*TWSS)


# Scree plot (Elbow Method) for choosing K visually
plt.plot(k_values_sorted, inertia_values, 'ro-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Total Within-Cluster Sum of Squares (TWSS)")
plt.title("Elbow Method Scree Plot")
plt.show()

# --------------------------
# Using KneeLocator for Best K Detection
# --------------------------

inertia_list = []  # To store inertia values

# Loop through the full K range for KneeLocator
for k in range(2, 9):
    kmeans = KMeans(n_clusters = k, init = "random", max_iter = 30, n_init = 10, random_state = 42)
    kmeans.fit(df_clean)
    inertia_list.append(kmeans.inertia_)

# Find the best K using KneeLocator
kl = KneeLocator(range(2, 9), inertia_list, curve = 'convex', direction = 'decreasing')
best_k_knee = kl.elbow  # Optimal K determined by the elbow point

# Print best K detected by KneeLocator
print("Best K (Knee Method):", best_k_knee)

# Plot Knee Method for visualizing the elbow point
plt.style.use("ggplot")
plt.plot(range(2, 9), inertia_list, marker='o', linestyle='-')
plt.xticks(range(2, 9))
plt.ylabel("Inertia")
plt.xlabel("Number of Clusters (K)")
plt.axvline(x=best_k_knee, color='r', linestyle='--', label=f'Elbow at K={best_k_knee}')
plt.legend()
plt.title("Knee Method for Optimal K")
plt.show()



# --------------------------
# Final KMeans Model with Optimal K
# --------------------------

# Set the final number of clusters to the best found K
final_k = best_k_knee if best_k_knee else best_k  # Prioritize KneeLocator result

# Create KMeans model with best K
final_model = KMeans(n_clusters = final_k, init = "k-means++", random_state = 42)

# Fit the model
final_model.fit(df_clean)

# Get cluster labels
cluster_labels = final_model.labels_

# Print final cluster assignments
print("Final Cluster Labels:", np.unique(cluster_labels))

# --------------------------
# Cluster Evaluation Metrics
# --------------------------
from sklearn import metrics

# Silhouette Score: Measures cohesion and separation
silhouette_score_value = metrics.silhouette_score(df_clean, final_model.labels_)
print("Silhouette Score:", silhouette_score_value)


# --------------------------
# Saving Model using Pickle
# --------------------------
pickle.dump(final_model, open('Clust_Univ.pkl', 'wb'))
print("Model saved successfully.")

# --------------------------
# Exporting Results
# --------------------------

# Obtaining cluster labels as a Pandas Series
mb = pd.Series(cluster_labels)

# Concatenating cluster labels with original data
df_clust = pd.concat([mb, df], axis = 1)
df_clust = df_clust.rename(columns = {0: 'cluster_id'})

# Display first few rows of the clustered data
print(df_clust.head())

# Aggregate data using mean for each cluster
cluster_agg = df_clust.iloc[:, 3:].groupby(df_clust.cluster_id).mean()
print(cluster_agg)

















