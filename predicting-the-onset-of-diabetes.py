#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Motivation to Classify or Predict Diabetes</font></h3>
#  
#    The classification of diabetes is a significant task in healthcare for several reasons, and the motivation for developing models to predict or classify diabetes can be understood from various perspectives:
# 
# - Early Detection and Intervention: Diabetes is a chronic condition that can lead to serious complications if not managed properly. Early detection of diabetes or prediabetes allows for timely intervention, lifestyle changes, and medical treatment. This can prevent or delay the onset of complications such as heart disease, kidney failure, blindness, and limb amputations.
# 
# - Personalized Care: Predictive models can help in identifying individuals at higher risk of developing diabetes based on their unique health profiles, genetics, lifestyle, and environmental factors. This enables healthcare providers to create personalized prevention and treatment plans, improving patient outcomes.
# 
# - Resource Allocation: Identifying individuals at high risk of developing diabetes helps healthcare systems allocate resources more efficiently. Targeted screening and prevention programs can be developed for those at higher risk, optimizing healthcare expenditures.
# 
# - Public Health Planning: Understanding the risk factors for diabetes and predicting its prevalence in different populations supports public health planning and policy development. It guides the implementation of prevention programs, awareness campaigns, and healthcare infrastructure planning.
# 
# - Research and Drug Development: Classification models can also aid in research by identifying potential biological markers or pathways associated with diabetes. This knowledge can fuel further research and lead to the development of new therapeutic interventions and drugs.
# 
# - Empowering Individuals: Tools that can predict or classify diabetes can be integrated into personal health applications and devices, empowering individuals to take control of their health. By understanding their risk factors, individuals can make informed lifestyle choices to reduce their risk of developing diabetes.
# 
# - Economic Impact: Diabetes is associated with significant economic burdens on healthcare systems. Effective prediction, prevention, and management can reduce hospitalizations, complications, and overall healthcare costs.
# 

# <a id="read"></a>
# # <p style="background-color:#c6edf1; font-family:calibri; color:#ff7034; font-size:100%; text-align:center; border-radius:10px 30px;margin:20; padding:10px;"> Import Dataset </p>

# In[1]:


#Reading the dataset
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df = pd.read_csv("../input/diabetes-data-set/diabetes.csv")
df


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Dataset Content</font></h3>
#     
# - Pregnancies: Number of times pregnant.
# - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
# - BloodPressure: Diastolic blood pressure (mm Hg).
# - SkinThickness: Triceps skin fold thickness (mm).
# - Insulin: 2-Hour serum insulin (mu U/ml).
# - BMI: Body mass index (weight in kg/(height in m)^2).
# - DiabetesPedigreeFunction
# - Age: Age (years).
# - Outcome: '1' denotes patient having diabetes and '0' denotes patient not having diabetes.

# <a id="read"></a>
# # <p style="background-color:#c6edf1; font-family:calibri; color:#ff7034; font-size:100%; text-align:center; border-radius:10px 30px;margin:20; padding:10px;"> Preprocessing Step</p>

# In[2]:


print(f'Duplicated rows are: \n {df.duplicated().sum()} \n \n \n Null values per column are: \n {df.isnull().sum()}\n \n \n Zero values per column are: \n {(df == 0).sum()} , \n \n \n data types of each column is: \n {df.dtypes} \n \n \n  statistical details of the dataset is: \n {df.describe()} ')


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Result of Preprocessing </font></h3>
#     
# - As can be seen in the preprocessing result, dataset doesn't have duplicated and missing values 
# 
# 
# - Statistical part of this dataset shows that 
# 
#  - Pregnancies: The average number of pregnancies is around 3.84 with a standard deviation of 3.37. The maximum number of pregnancies recorded is 17.
#  - Glucose: The average glucose level is 120.89 with a standard deviation of 31.97. The minimum value is 0 which is not medically possible and it shows a missing or incorrect data.
#  - BloodPressure: The average blood pressure is around 69.10 with a standard deviation of 19.36. Similar to glucose, a blood pressure of 0 is not possible and indicates missing or incorrect data.
#  - SkinThickness: The average skin thickness is around 20.54 with a standard deviation of 15.95. There are also records with skin thickness of 0, which indicates missing or incorrect data.
#  - Insulin: The average insulin level is around 79.80 with a standard deviation of 115.24. Records with insulin level of 0 also indicate missing or incorrect data.
#  - BMI: The average BMI is around 31.99 with a standard deviation of 7.88. A BMI of 0 is not possible and indicates missing or incorrect data.
#  - DiabetesPedigreeFunction: The average value is around 0.47 with a standard deviation of 0.33.
#  - Age: The average age is around 33.24 with a standard deviation of 11.76.
#  - Outcome: About 34.9% of the patients in the dataset have diabetes.

# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Check for Suspicious Rows</font></h3>
# 
# Check how many rows have impossible values for Glucose, BloodPressure, SkinThickness, Insulin, and BMI

# In[3]:


impossible_values = (df["Glucose"] == 0) | (df["BloodPressure"] == 0) | (df["SkinThickness"] == 0) | (df["Insulin"] == 0) | (df["BMI"] == 0)
impossible_values.sum()


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# the result shows that we have 376 values that they have been recorded by mistake or they are missing values , so to revise this dataset we should fill these zeros with median of dataset. means first we replace 0 with NAN or not a value and then replace nan with median.

# In[4]:


import numpy as np

lst=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for i in lst:
    df[i].replace(0, np.nan, inplace=True)
    df[i].fillna(df[i].median(), inplace=True)
df.head()


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Visualizitaion of Distribution</font></h3>
# After completing preprocessing steps, we can investigate the distriubtion of every columns in dataset
# 

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
lst1=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
lst_col=['skyblue','olive','gold','teal','brown','purple','green','red','blue']
f, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=False) # Set up the matplotlib figure
axes = axes.flatten()  # Plot a simple histogram with binsize determined automatically
for ax,k,m in zip(axes,lst1,lst_col):
    ax.hist(df[k], color=m, bins=10, alpha=0.5)
    sns.distplot(df[k], color=m, ax=ax)
plt.tight_layout() 


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Visualization of Outliers </font></h3>
#     
# Then we use box plots to detect potential outliers. 
#     
# The dots outside the whiskers in each boxplot represent potential outliers.

# In[6]:


# Generate a box plot for each feature
lst1=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
lst_col=['skyblue','olive','gold','teal','brown','purple','green','red','blue']
f, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=False) # Set up the matplotlib figure
axes = axes.flatten()  # Plot a simple histogram with binsize determined automatically
for ax,k,m in zip(axes,lst1,lst_col):
    sns.boxplot(data=df, x=k, color=m, ax=ax)
plt.tight_layout()  
plt.show()


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Check the Relationship Between Variables </font></h3>
# 
# Then we wanna check the relationships between the different variables(columns). This can provide insights into which variables are strongly or weakly associated with each other.

# In[7]:


corr = df.corr() # Compute the correlation matrix
mask = np.triu(np.ones_like(corr, dtype=bool)) # Generate a mask for the upper triangle
f, ax = plt.subplots(figsize=(11, 9)) # Set up the matplotlib figure
cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging colormap
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True) # Draw the heatmap with the mask and correct aspect ratio
plt.tight_layout()


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> What Correlation shows?</font></h3>
#     
# - The correlation coefficient values range from -1 to 1. If the correlation coefficient is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation.
# - Glucose, Age and BMI are moderately correlated with Outcome.
# - Pregnancies and Age show a strong correlation.

# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Pairwise Variable Relationship </font></h3>
#     
# To understand pairwise relationships we  use pair plots
# - This can be very helpful to understand how the variables interact with each other and identify any potential patterns or trends in the data.

# In[8]:


# We'll use a sample of the data to make the pairplot faster to generate
df_sample = df.sample(100, random_state=1)

# Create a pairplot
sns.pairplot(df_sample, hue="Outcome")
plt.show()


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Split Dataset </font></h3>
#  Sure, let's start by applying some regression algorithms to predict the Outcome variable. Here, I will use Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines. Afterward, we will compare the performance of these models based on their accuracy.
# 
# Please note that the choice of accuracy as a metric here is for simplicity and because our classes are not heavily imbalanced. In real-world applications, especially for medical datasets, precision, recall, the F1 score, and the ROC AUC score are often more informative metrics, particularly in the case of imbalanced classes.
# 
# Before we start, we need to split the data into a training set and a test set. This allows us to evaluate how well our model generalizes to unseen data. We'll use 80% of the data for training and 20% for testing.
# 
# 

# In[9]:


#classification based on outcome column
#we need to split the data into a training set and a test set.
#This allows us to evaluate how well our model generalizes to unseen data.
#We'll use 80% of the data for training and 20% for testing.
from sklearn.model_selection import train_test_split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Split the data into a training set and a test set
X_train.shape, X_test.shape


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Scaling Data </font></h3>
#  We've split our data into a training set with 614 samples and a test set with 154 samples.
#  Now, let's scale the features. This is not necessary for all algorithms (for example, decision trees and random forest don't require scaling), but it doesn't hurt them and it is beneficial for others (like logistic regression and SVM). We'll use StandardScaler from scikit-learn to standardize the features to have mean=0 and variance=1.

# In[10]:


#now we wanna scale our features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # Initialize the scaler
scaler.fit(X_train) # Fit the scaler to the training data
X_train_scaled = scaler.transform(X_train) # Transform the training and test sets
X_test_scaled = scaler.transform(X_test)
X_train_scaled.shape, X_test_scaled.shape


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Training Phase </font></h3>
# We have successfully scaled our features. Now, let's train and evaluate the different models.
# We'll create a function to fit a model and evaluate its accuracy on both the training set and the test set. It will help to reduce the amount of repetitive code. Let's define this function and then use it to evaluate the logistic regression, decision tree, random forest, and SVM models.

# In[11]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
def fit_and_evaluate(model):# Define a function to fit a model 
    model.fit(X_train_scaled, y_train)  # Fit the model
    train_preds = model.predict(X_train_scaled) # Make predictions on the training set and compute the accuracy
    train_acc = accuracy_score(y_train, train_preds)
    test_preds = model.predict(X_test_scaled) # Make predictions on the test set and compute the accuracy
    test_acc = accuracy_score(y_test, test_preds)
    return train_acc, test_acc
res=[]
models=[LogisticRegression(random_state=42),DecisionTreeClassifier(random_state=42),RandomForestClassifier(random_state=42),SVC(random_state=42)]
for i in models:
    res.append(fit_and_evaluate(i))
for mod,result in zip(models,res):
    print(f'train and test accuracy of {mod} respectively is: \n {result} \n \n')


# <div style="border-radius:10px; padding: 15px; background-color: #c6edf1; font-size:115%; text-align:left">
# <h3 align="left"><font color=#ff7034> Result & Discussion  </font></h3>
#     
# - Logistic Regression: The training accuracy is approximately 77.04% and the test accuracy is approximately 75.32%.
# - Decision Tree: The training accuracy is 100%, indicating that the model has perfectly fit the training data. However, the test accuracy drops to approximately 71.43%, suggesting that the model might be overfitting the training data.
# - Random Forest: Similar to the Decision Tree, the training accuracy is 100%, but the test accuracy is approximately 73.38%, indicating potential overfitting.
# - Support Vector Machine (SVM): The training accuracy is approximately 82.90% and the test accuracy is approximately 74.68%.
# 
# From the results, it appears that the Logistic Regression and SVM models are performing the best on this dataset based on accuracy. They are not overfitting the training data as much as the Decision Tree and Random Forest models, and are achieving a good balance between bias and variance.
#     
# - NOTE: keep in mind that these models have been run with default parameters. Tuning the hyperparameters could potentially improve their performance.
# 
# 
