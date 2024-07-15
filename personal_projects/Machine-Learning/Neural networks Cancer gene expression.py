#!/usr/bin/env python
# coding: utf-8

# 
# ## Introduction
# In this project, we will explore the application of machine learning, specifically neural networks, for cancer prediction using gene expression data. Cancer is a complex and multifaceted disease, and early detection is crucial for effective treatment and improved patient outcomes. By leveraging gene expression data, we can identify patterns and biomarkers that are indicative of various types of tumors, potentially leading to better diagnostic tools and therapeutic strategies.
# 
# ### Dataset Description
# The dataset used in this project is part of the RNA-Seq (HiSeq) PANCAN data set, which includes gene expression data from patients with different types of tumors: breast cancer (BRCA), kidney renal clear cell carcinoma (KIRC), colon adenocarcinoma (COAD), lung adenocarcinoma (LUAD), and prostate adenocarcinoma (PRAD). The data was donated on June 8, 2016, and is stored in a multivariate format, suitable for tasks such as classification and clustering.
# 
# ### Key Characteristics:
# - Subject Area: Biology 
# - Associated Tasks: Classification, Clustering
# - Feature Type: Real
# - Number of Instances: 801
# - Number of Features: 20,531
# - Missing Values: None
# 
# Each instance (sample) in the dataset corresponds to a patient, with gene expression levels measured by the Illumina HiSeq platform. The attributes are named using a dummy naming convention (gene_XX), consistent with the original submission of the dataset. Detailed probe names and platform specifications can be found on the Synapse website.
# 
# ### Project Objectives
# The primary objective of this project is to build a neural network model capable of predicting the type of tumor based on the gene expression profiles of patients. Therefore the most important metric in this model is **Accuracy**. The steps involved in achieving this objective are as follows:
# 
# ### Data Preprocessing
# Load the dataset and perform initial exploration.
# Normalize the gene expression data to ensure that all features contribute equally to the model.
# Split the data into training and testing sets.
# 
# ### Model Development
# Design a neural network architecture suitable for high-dimensional gene expression data.
# Train the neural network on the training set, optimizing for accuracy and generalization.
# 
# ### Model Evaluation
# Evaluate the trained model on the testing set using accuracy. 
# 
# ### Results Interpretation and Visualization
# Analyze the model's performance and identify the most important genes contributing to the predictions.
# Visualize the results using confusion matrices, ROC curves, and other relevant plots.
# 
# ### Conclusion and Future Work
# Summarize the findings and discuss the implications for cancer prediction and potential clinical applications.
# Suggest improvements and future directions for further research.
# 
# By the end of this project, we aim to demonstrate the potential of neural networks in predicting cancer types from gene expression data, highlighting the importance of machine learning in advancing biomedical research and improving patient care.
# 
# Dataset Link = https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

# In[1]:


get_ipython().system('pip install tensorflow==2.15.0 scikit-learn==1.2.2 matplotlib===3.7.1 seaborn==0.13.1 numpy==1.25.2 pandas==1.5.3 -q --user')


# In[2]:


# Library for data manipulation and analysis.
import pandas as pd
# Fundamental package for scientific computing.
import numpy as np
#splitting datasets into training and testing sets.
from sklearn.model_selection import train_test_split
#Imports tools for data preprocessing including label encoding, one-hot encoding, and standard scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
#Imports a class for imputing missing values in datasets.
from sklearn.impute import SimpleImputer
#Imports the Matplotlib library for creating visualizations.
import matplotlib.pyplot as plt
# Imports the Seaborn library for statistical data visualization.
import seaborn as sns
# Time related functions.
import time
#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report


#Imports the tensorflow,keras and layers.
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout,BatchNormalization
from tensorflow.keras import backend

# to suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")


# In[3]:


#Data handling
import pandas as pd
import numpy as np

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#Classification
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#To suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")


# In[4]:


#Read data directly from a github repository
file_url='https://github.com/AhmedShehata2002/Personal_Projects/raw/main/Data/cancer_gene_expression.zip'
dataframe=pd.read_csv(file_url)


# In[5]:


dataframe


# In[6]:


dataframe.head()


# In[7]:


dataframe.info()


# In[8]:


#Let's check for any duplicates in the data
dataframe.duplicated().sum()


# In[9]:


#Lets check for any missing values in the data 
datanul=dataframe.isnull().sum()
g=[i for i in datanul if i>0]

print('columns with missing values:%d'%len(g))


# In[146]:


cancer_type_counts= dataframe['Cancer_Type'].value_counts()
cancer_type_counts


# In[147]:


plt.figure(figsize=(8, 5))
cancer_type_counts.plot(kind='bar', color='skyblue')
plt.title('Cancer Type Distribution')
plt.xlabel('Cancer Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[11]:


print(dataframe.columns[0:3])


# In[12]:


#Checking the name of the last column
dataframe.columns[-1]


# ## Data Preprocessing

# Firstly we will have to separate the feature values from the class.  

# In[13]:


X=dataframe.iloc[:,0:-1]
y=dataframe.iloc[:,-1]


# ### Encoding Labels 
# Then we will have to encode and change the categorical variables to numerical values. 

# In[14]:


label_encoder=LabelEncoder()
label_encoder.fit(y)
y=label_encoder.transform(y)
labels=label_encoder.classes_
classes=np.unique(y)
nclasses=np.unique(y).shape[0]


# ### Data Splitting 

# In[15]:


#Split the data into training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Split the training set into two (training and validation)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2)


# ### Data Normalisation 

# In[16]:


min_max_scaler=MinMaxScaler()
X_train=min_max_scaler.fit_transform(X_train)
X_val=min_max_scaler.fit_transform(X_val)
X_test=min_max_scaler.fit_transform(X_test)


# ### Neural Network with SGD Optimizer

# In[123]:


tf.keras.backend.clear_session()


# In[124]:


#Defining the model
model = Sequential()

#Hidden layer 1
model.add(Dense(40, input_dim=X_train.shape[1], activation='relu'))

#Hidden layer 2
model.add(Dense(20, activation='relu'))

#Output layer
model.add(Dense(nclasses, activation='softmax'))

#Define optimizer and learning rate. We will use SDG optimizer
opt_sdg = keras.optimizers.SGD(learning_rate=0.001)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_sdg, metrics=[keras.metrics.SparseCategoricalAccuracy()])


# In[125]:


model.summary()


# In[126]:


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,epochs=200, verbose=1)


# In[127]:


predictions = model.predict(X_test)
accuracy = model.evaluate(X_train, y_test, verbose=0)


# In[128]:


#Now we will print the predictions for the first 20 samples in the test set
for index,entry in enumerate(predictions[0:20,:]):
    print('predicted:%d ,actual:%d'%(np.argmax(entry),y_test[index]))


# In[129]:


# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


# In[130]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


# In[131]:


label_encoder = LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)


# In[132]:


nclasses = len(np.unique(y_train))  # Assuming y_train is your training labels


# In[133]:


labels = label_encoder.classes_  # Array of original class labels
classes = np.unique(y_encoded)   # Unique encoded class labels
nclasses = classes.shape[0]      # Number of unique classes


# In[134]:


# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# ### Observations:
# As we can see the model performance could be much better, I belive by switching to an Adam Optimizer a lot of these inconsistencies will be solved. 

# ## Neural network with Adam Optimizer

# In[135]:


tf.keras.backend.clear_session()


# In[136]:


#Defining the model
model = Sequential()

#Hidden layer 1
model.add(Dense(40, input_dim=X_train.shape[1], activation='relu'))

#Hidden layer 2
model.add(Dense(20, activation='relu'))

#Output layer
model.add(Dense(nclasses, activation='softmax'))

#Define optimizer and learning rate. We will use Adam optimizer
opt_adam = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_adam, metrics=[keras.metrics.SparseCategoricalAccuracy()])


# In[137]:


model.summary()


# In[138]:


#Fit the model to the training data
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,epochs=200, verbose=1)


# In[139]:


#Now we will print the predictions for the first 20 samples in the test set
for index,entry in enumerate(predictions[0:20,:]):
    print('predicted:%d ,actual:%d'%(np.argmax(entry),y_test[index]))


# As we can see that the predicted and actual data are all the same therefore this neural network is work really well. 

# In[140]:


# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


# In[141]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


# In[142]:


# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# ## Conclusion:

# This project involved training a neural network to perform classification tasks using two different optimizers: Adam and SGD (Stochastic Gradient Descent). The architecture of the neural network was consistent across both experiments, allowing for a direct comparison of the optimizer performance. The results indicate that the model trained with the Adam optimizer outperformed the model trained with the SGD optimizer in terms of both loss and accuracy.
# 
# 
# ### Prediction Consistency:
# Both models showed consistent predictions aligning closely with the actual labels, but the Adam optimizer achieved this with higher precision.
# 
# ### Conclusion on Model Performance
# The model trained with the Adam optimizer is clearly better than the model trained with the SGD optimizer. The Adam optimizer led to a higher test accuracy and a significantly lower test loss, suggesting that it is more effective in optimizing the neural network for this particular classification task.
# 
# ### Areas for Further Improvement
# - Hyperparameter Tuning:
# Further tuning of hyperparameters such as learning rate, batch size, and number of epochs could potentially enhance the performance of both models.
# 
# - Regularization Techniques:
# Incorporating regularization techniques like dropout, L2 regularization, or early stopping might help in preventing overfitting and improving generalization.
# 
# 
# - Optimizer Variants:
# Experimenting with other optimizers or hybrid optimization techniques might yield better results, especially for specific datasets or tasks.
# 
# - Model Architecture:
# Exploring different model architectures, including deeper networks or different layer configurations, could lead to performance improvements.
# 
# In conclusion, while the Adam optimizer proved superior in this project, further experimentation and optimization can help in achieving even better results and potentially discovering new insights.
