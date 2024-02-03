#!/usr/bin/env python
# coding: utf-8

# # Data Science Task 1
# 
# ## Task: Detecting Fake News
# 
# ## Objective: Differentiate between real and fake news using a Python project applying a PassiveAggressiveClassifier.

# ### Tasks:
# 
# ### 1. Read and explore the textual dataset.
# ### 2. Build a machine learning model with TfidfVectorizer and PassiveAggressiveClassifier.
# ### 3. Create a confusion matrix to evaluate the model's performance.
# ### 4. Measure the model's accuracy.

# ### Steps
# 
# ### 1- Import necessary libraries
# ### 2- Read and explore the dataset
# ### 3- Build a model using PassiveAggressiveClassifier
# ### 4- Evaluate the model's accuracy

# In[ ]:


pip install numpy pandas sklearn


# #### 1- Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


# #### 2- Read and explore the dataset

# In[ ]:


news_data= pd.read_csv("dataset/news.csv")
news_data.head(10)


# In[ ]:


news_data.info()


# In[ ]:


news_data.shape


# In[ ]:


news_data["label"].value_counts()


# In[ ]:


labels= news_data.label
labels.head(10)


# ### 1st model 

# #### 3- Build the model

# In[ ]:


#First, we split the dataset into train & test samples:
x_train, x_test, y_train, y_test= train_test_split(news_data["text"], labels, test_size= 0.2, random_state= 7)


# In[ ]:


#Then we’ll initialize TfidfVectorizer with English stop words
vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=vectorizer.fit_transform(x_train)
tfidf_test=vectorizer.transform(x_test)


# In[ ]:


#Create a PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train,y_train)

y_pred=passive.predict(tfidf_test)


# #### 4- Evaluate the model's accuracy

# In[ ]:


#Create a confusion matrix
matrix= confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
matrix


# In[ ]:


#Visualize the confusion matrix
sns.heatmap(matrix, annot=True)
plt.show()


# In[ ]:


#Calculate the model's accuracy
Accuracy=accuracy_score(y_test,y_pred)
Accuracy*100


# In[ ]:


## The model's accuracy is 93%
Report= classification_report(y_test, y_pred)
print(Report)


# In[ ]:


#Now let’s test this model. 
#To test our trained model, I’ll first write down the title of any news item found on google news to see if our 
#model predicts that the news is real or not:

news_headline_1 = "Trump takes on Cruz, but lightly"

data = vectorizer.transform([news_headline_1]).toarray()
print(passive.predict(data))


# In[ ]:


#Now I’m going to write a random fake news headline to see if the model predicts the news is fake or not:

news_headline_2 = "Cow dung can cure Corona Virus"
data = vectorizer.transform([news_headline_2]).toarray()
print(passive.predict(data))


# In[ ]:


news_headline_3 = "Doubt Congress will get ‘even 40 seats’ in LS polls, says Mamata"
data = vectorizer.transform([news_headline_3]).toarray()
print(passive.predict(data))


# ### 2nd Model to Increase Accuracy

# #### 3- Build the model

# In[ ]:


#First, we split the dataset into train & test samples:
x_train,x_test,y_train,y_test=train_test_split(news_data['text'], labels, test_size=0.4, random_state=7)


# In[ ]:


vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
## fit and transform train set, transform test set
tfidf_train=vectorizer.fit_transform(x_train)
tfidf_test=vectorizer.transform(x_test)


# In[ ]:


#Create a PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train,y_train)

y_pred=passive.predict(tfidf_test)


# #### 4- Evaluate the model's accuracy

# In[ ]:


#Create a confusion matrix
matrix= confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
matrix


# In[ ]:


#Visualize the confusion matrix
sns.heatmap(matrix, annot=True)
plt.show()


# In[ ]:


#Calculate the model's accuracy
Accuracy=accuracy_score(y_test,y_pred)
Accuracy*100


# In[ ]:


## The model's accuracy is 93%
Report= classification_report(y_test, y_pred)
print(Report)


# In[ ]:


#Now let’s test this model. 
#To test our trained model, I’ll first write down the title of any news item found on google news to see if our 
#model predicts that the news is real or not:

news_headline_1 = "Trump takes on Cruz, but lightly"

data = vectorizer.transform([news_headline_1]).toarray()
print(passive.predict(data))


# In[ ]:


#Now I’m going to write a random fake news headline to see if the model predicts the news is fake or not:

news_headline_2 = "Cow dung can cure Corona Virus"
data = vectorizer.transform([news_headline_2]).toarray()
print(passive.predict(data))


# In[ ]:


news_headline_3 = "Doubt Congress will get ‘even 40 seats’ in LS polls, says Mamata"
data = vectorizer.transform([news_headline_3]).toarray()
print(passive.predict(data))


# ### 3rd Model to further increase accuracy

# #### 3- Build the model

# In[ ]:


#First, we split the dataset into train & test samples:
x_train,x_test,y_train,y_test=train_test_split(news_data['text'], labels, test_size=0.3, random_state=6)


# In[ ]:


vectorizer=TfidfVectorizer(stop_words='english', max_df=0.9)
## fit and transform train set, transform test set
tfidf_train=vectorizer.fit_transform(x_train)
tfidf_test=vectorizer.transform(x_test)


# In[ ]:


#Create a PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train,y_train)

y_pred=passive.predict(tfidf_test)


# #### 4- Evaluate the model's accuracy

# In[ ]:


#Create a confusion matrix
matrix= confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
matrix


# In[ ]:


#Visualize the confusion matrix
sns.heatmap(matrix, annot=True)
plt.show()


# In[ ]:


#Calculate the model's accuracy
Accuracy=accuracy_score(y_test,y_pred)
Accuracy*100


# In[ ]:


## The model's accuracy is 93.52%
Report= classification_report(y_test, y_pred)
print(Report)


# In[ ]:


#Now let’s test this model. 
#To test our trained model, I’ll first write down the title of any news item found on google news to see if our 
#model predicts that the news is real or not:

news_headline_1 = "Trump takes on Cruz, but lightly"

data = vectorizer.transform([news_headline_1]).toarray()
print(passive.predict(data))


# In[ ]:


#Now I’m going to write a random fake news headline to see if the model predicts the news is fake or not:

news_headline_2 = "Cow dung can cure Corona Virus"
data = vectorizer.transform([news_headline_2]).toarray()
print(passive.predict(data))


# In[ ]:


news_headline_3 = "Doubt Congress will get ‘even 40 seats’ in LS polls, says Mamata"
data = vectorizer.transform([news_headline_3]).toarray()
print(passive.predict(data))


# ### 4th Model to Increase Accuracy

# #### 3- Build the model

# In[ ]:


#First, we split the dataset into train & test samples:
x_train,x_test,y_train,y_test=train_test_split(news_data['text'], labels, test_size=0.2, random_state=10)


# In[ ]:


vectorizer=TfidfVectorizer(stop_words='english', max_df=0.9)
## fit and transform train set, transform test set
tfidf_train=vectorizer.fit_transform(x_train)
tfidf_test=vectorizer.transform(x_test)


# In[ ]:


#Create a PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train,y_train)

y_pred=passive.predict(tfidf_test)


# #### 4- Evaluate the model's accuracy

# In[ ]:


#Create a confusion matrix
matrix= confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
matrix


# In[ ]:


#Visualize the confusion matrix
sns.heatmap(matrix, annot=True)
plt.show()


# In[ ]:


#Calculate the model's accuracy
Accuracy=accuracy_score(y_test,y_pred)
Accuracy*100


# In[ ]:


## The model's accuracy is 94.86%
Report= classification_report(y_test, y_pred)
print(Report)


# In[ ]:


#Now let’s test this model. 
#To test our trained model, I’ll first write down the title of any news item found on google news to see if our 
#model predicts that the news is real or not:

news_headline_1 = "Trump takes on Cruz, but lightly"

data = vectorizer.transform([news_headline_1]).toarray()
print(passive.predict(data))


# In[ ]:


#Now I’m going to write a random fake news headline to see if the model predicts the news is fake or not:

news_headline_2 = "Cow dung can cure Corona Virus"
data = vectorizer.transform([news_headline_2]).toarray()
print(passive.predict(data))


# In[ ]:


news_headline_3 = "Doubt Congress will get ‘even 40 seats’ in LS polls, says Mamata"
data = vectorizer.transform([news_headline_3]).toarray()
print(passive.predict(data))

