#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Classification with Numeric and Categorical Attributes: Exploring Cost Matrix Impact    

# - Project by: Ahmed Shehata 
# - e-mail:ahmed1979fcb@gmail.com
# - Linkedin:www.linkedin.com/in/shehata-ahmed
# - Website: https://ahmedshehata2002.github.io/Portfolio-/

# ## Objective 

# In this project, we aim to develop a predictive model using machine learning classification techniques to assess whether individuals seeking loans pose a risk of defaulting or not. Leveraging a dataset containing a variety of independent variables such as checking account balance, credit history, purpose, and loan amount, our objective is to create a robust model that accurately categorizes loan applicants as either defaulters or non-defaulters. Through the utilization of both numeric and categorical attributes, we seek to enhance the predictive capabilities of our model. Additionally, the exploration of a cost matrix will allow us to analyze the impact of misclassification costs on model performance and refine our classification strategy for improved accuracy and reliability in credit risk assessment. Data from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

# ### Key Steps in Developing Decision Tree Models for Credit Risk Classification

# 1. Data Summary: Initially, we'll summarize the dataset, exploring its structure, dimensions, and the nature of its variables. This step involves understanding the dataset's composition and identifying any missing values or outliers that may require preprocessing.
# 
# 2. Univariate and Bivariate Analysis: Following data summary, we'll conduct univariate analysis to explore individual variables' distributions and characteristics. Then, we'll proceed to bivariate analysis to examine relationships between pairs of variables and uncover potential patterns or correlations.
# 
# 3. Exploratory Data Analysis (EDA): EDA involves a comprehensive exploration of the dataset using statistical techniques and data visualization methods. This step helps us gain deeper insights into the data, identify trends, patterns, and anomalies, and inform subsequent modeling decisions.
# 
# 4. Training the Data: After EDA, we'll prepare the data for model training by splitting it into training and testing sets. The training set will be used to train our machine learning models, while the testing set will evaluate model performance and generalization to unseen data.
# 
# 5. Ensemble Model Creation: Leveraging bagging and boosting techniques alongside ensemble methods such as Bagging Classifiers, we construct ensemble models to enhance predictive accuracy. These models harness the collective wisdom of diverse algorithms, mitigating individual model biases and enhancing robustness.
# 
# 6. Visualization of Ensemble Models: Visualizing the ensemble models elucidates their collective decision-making process and structural nuances. This visualization aids in comprehending how the models amalgamate diverse perspectives to make accurate predictions.
# 
# 7. Assessment of Feature Importances using GINI: Finally, we'll assess the importance of features in our decision tree models using the GINI impurity criterion. GINI impurity measures the extent of class impurity within a node, and feature importances derived from GINI provide valuable insights into the predictive power of each feature in the model.
# 
# 8. Pipeline Modelling: Preceding our recommendations, we acknowledge the criticality of operationalizing the developed model effectively. To ensure seamless integration into real-world applications, we prioritize building a robust data engineering pipeline. This pipeline will streamline data preprocessing, model training, and deployment processes, facilitating efficient utilization of the ensemble model for credit risk classification tasks. Establishing a well-defined pipeline is fundamental for maximizing the model's impact and scalability, while maintaining data integrity and reliability throughout the workflow.
# 
# 9. Final Model Summary, Recommendations, and Conclusion: Culminating our efforts, we present a comprehensive overview of the final ensemble model. We elucidate performance metrics, feature importances, and actionable insights derived from the analysis. Our recommendations inform credit risk assessment practices, encapsulating key takeaways from the modeling journey.
# 
# 
# By following these steps systematically, we aim to develop a robust decision tree model for credit risk classification, leveraging exploratory analysis and feature importance assessment to enhance model interpretability and predictive performance.

# In[249]:


get_ipython().system(' pip install black')


# In[3]:


# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to split data and manipulate it 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# to build linear regression_model
from sklearn.linear_model import LinearRegression
# to check model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# to build linear regression_model using statsmodels
import statsmodels.api as sm
# to compute VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[4]:


# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

# To help with model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

# To get different metric scores, and split data
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


# In[5]:


# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# To be used for tuning the model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To define maximum number of columns to be displayed in a dataframe
pd.set_option("display.max_columns", None)

# To supress scientific notations for a dataframe
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To supress warnings
import warnings

warnings.filterwarnings("ignore")

# This will help in making the Python code more structured automatically (good coding practice)
#%load_ext nb_black


# In[6]:


df = pd.read_csv('credit.csv')


# ## Summary of the data 

# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# ## Missing Values 

# In[10]:


df.isnull().values.any()


# - No missing values in dataset 

# In[258]:


df.isnull().sum()


# ## Duplicate Values 

# In[11]:


df.duplicated().sum()


# In[259]:


cat_col = [
    "checking_balance",
    "credit_history",
    "purpose",
    "savings_balance",
    "employment_duration",
    "other_credit",
    "housing",
    "job",
    "phone",
    "default"]


# In[260]:


for column in cat_col:
    print(df[column].value_counts())
    print("-" * 40)


# ### Observations: 
# - A lot of the data types are objects and in order to work with it in the classification model they would need to be convereted into integers. After conducting the Univariate and Bivariate analysis I will need to transform the data. 
# 

# ### Univariate analysis

# In[261]:


num_cols=df.select_dtypes(include=np.number).columns.tolist()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
for num_col in num_cols:
  sns.histplot(data=df,x=num_col,ax=axes[1],kde=True)
  plt.show()
  sns.boxplot(data=df,x=num_col)
  plt.show


# ### Observations:
# - There are quite a few outliers in the data
# - However, they look like proper values

# ## Numerical Data 

# In[262]:


def histogram_boxplot(df, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined


    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=df, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=df, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=df, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        df[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        df[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[263]:


histogram_boxplot(df,"months_loan_duration")


# In[264]:


histogram_boxplot(df,"age")


# In[265]:


histogram_boxplot(df,"amount")


# ## Categorical Data

# In[266]:


def labeled_barplot(df, feature, perc=False, n=None):
    """
    Barplot with percentage at the top


    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """


    total = len(df[feature])  # length of the column
    count = df[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))


    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=df,
        x=feature,
        palette="Paired",
        order=df[feature].value_counts().index[:n],
    )


    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category


        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot


        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage


    plt.show()  # show the plot


# In[267]:


labeled_barplot(df, "default")


# In[268]:


labeled_barplot(df, "savings_balance")


# In[269]:


labeled_barplot(df, "purpose")


# In[270]:


labeled_barplot(df,"checking_balance")


# In[271]:


labeled_barplot(df,"credit_history")


# In[272]:


labeled_barplot(df,"savings_balance")


# In[273]:


labeled_barplot(df,"employment_duration")


# In[274]:


labeled_barplot(df,"other_credit")


# In[275]:


labeled_barplot(df,"housing")


# In[276]:


labeled_barplot(df,"job")


# ### Observations: 
# - **Risk Distribution**: The dataset shows an imbalanced class distribution in the target variable, with 70% non-defaulters and 30% defaulters, indicating a potential need for oversampling or other techniques to address the class imbalance.
# - **Housing Ownership**: The majority of customers (71%) who take credit own their houses, followed by 18% living in rented accommodations, and only 11% with free housing provided by their employers, indicating the significance of housing status in credit-seeking behavior.
# - **Job Categories**: Most customers (63%) fall into the skilled job category, with only approximately 15% categorized as highly skilled, suggesting that job type may influence creditworthiness and borrowing behavior.
# - **Purpose of Credit**: The analysis reveals that most customers take credit for luxury items like cars, radios, furniture/equipment, and domestic appliances, while only around 16% take credit for business or education purposes, indicating consumer-centric rather than investment-driven borrowing patterns.

# ## Bivariate Analysis

# ### This analysis will allow us to look deeper and see some of the reasons some associations with defaulting 
# - Any variations will be useful to build a better model. 

# In[277]:


def distribution_plot_wrt_target(df, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = df[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=df[df[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=df[df[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=df, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=df,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


# In[278]:


distribution_plot_wrt_target(df, "months_loan_duration", "default")


# In[279]:


distribution_plot_wrt_target(df, "age", "default")


# In[280]:


distribution_plot_wrt_target(df, "amount", "default")


# In[281]:


distribution_plot_wrt_target(df, "existing_loans_count", "default")


# ### Observations:

# - **Age and Risk** : The median age of defaulters is lower than that of non-defaulters, suggesting that younger customers are more prone to defaulting. Additionally, there are outliers in the age distribution of both classes, indicating potential outliers affecting the analysis.
# - **Credit Amount and Risk** : Defaulters tend to have higher third quartile credit amounts compared to non-defaulters, implying that customers with larger credit amounts are more likely to default. Again, both classes exhibit outliers in credit amount distributions.
# - **Duration of Credit and Risk**: Defaulters tend to have longer durations for credit compared to non-defaulters, particularly evident in the second and third quartiles. This suggests that customers with longer credit durations are more prone to defaulting.
# - **Saving Accounts and Age**: Customers with higher ages tend to fall into the rich or quite rich category in terms of saving accounts. However, there are outliers in both the little and moderate saving account categories, indicating exceptions to this trend.

# ### Converting Categorical Data to Numerical 
# - This is essential firstly to be able to deprive more insights from the EDA and secoundly as this will be essential in classification. 

# In[282]:


df2 = df.copy()


# ### Note
# In order to replace the variables appropriately I will need to assess the rankings of each one. e.g the credit history would need to be assessed based on a rank instead of amount in data set. So it would be {"critical": 1, "poor":2 , "good": 3, "very good": 4,"perfect": 5}, NOT { 1.good:530, 2.critical 293,3.poor 88} etc.

# In[283]:


replaceStruct = {
                "checking_balance":     {"< 0 DM": 1, "1 - 200 DM": 2 ,"> 200 DM": 3 ,"unknown":-1},
                "credit_history": {"critical": 1, "poor":2 , "good": 3, "very good": 4,"perfect": 5},
                 "savings_balance": {"< 100 DM": 1, "100 - 500 DM":2 , "500 - 1000 DM": 3, "> 1000 DM": 4,"unknown": -1},
                 "employment_duration":     {"unemployed": 1, "< 1 year": 2 ,"1 - 4 years": 3 ,"4 - 7 years": 4 ,"> 7 years": 5},
                "phone":     {"no": 1, "yes": 2 },
                #"job":     {"unemployed": 1, "unskilled": 2, "skilled": 3, "management": 4 },
                "default":     {"no": 0, "yes": 1 } 
                    }


# In[284]:


df2


# Some columns although numerical but would still benefit from one hot encoding, this would enable the model to learn more effectively and improve classification accuracy. 

# In[285]:


oneHotCols=["purpose","housing","other_credit","job"]


# In[286]:


df2=df2.replace(replaceStruct)
df2=pd.get_dummies(df2, columns=oneHotCols)


# In[287]:


df2.info()


# ### Now that all the Dtypes are numerical we are able to start training the model. 

# ## Train and Split the Data 

# In[288]:


X = df2.drop("default" , axis=1)
y = df2.pop("default")


# In[289]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[290]:


# Splitting data into training, validation and test sets:
# first we split data into 2 parts, say temporary and test

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# then we split the temporary set into train and validation

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.3, random_state=1, stratify=y_temp
)
print(X_train.shape, X_val.shape, X_test.shape)


# In[291]:


# Creating dummy variables for categorical variables
X_train = pd.get_dummies(data=X_train, drop_first=True)
X_val = pd.get_dummies(data=X_val, drop_first=True)
X_test = pd.get_dummies(data=X_test, drop_first=True)


# ## Model Building 

# In[292]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf


# In[293]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# In[294]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("GBM", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))
models.append(("dtree", DecisionTreeClassifier(random_state=1)))

results = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models
score = []
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")
for name, model in models:
    scoring = "recall"
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scoring, cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean() * 100))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_val, model.predict(X_val))
    score.append(scores)
    print("{}: {}".format(name, scores))


# In[295]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure()

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results)
ax.set_xticklabels(names)

plt.show()


# - We can see that the decision tree is giving the highest cross-validated recall followed by xgboost
# - The boxplot shows that the performance of decision tree and xgboost is consistent and their performance on the validation set is also good
# - We will tune the best two models i.e. decision tree and xgboost and see if the performance improves

# ## Hyperparameter Tuning

# In[296]:


# Creating pipeline
model = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [2, 4, 7, 10, 15],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5)

# Fitting parameters in GridSeachCV
grid_cv.fit(X_train, y_train)

print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)


# In[297]:


# Creating new pipeline with best parameters
dtree_tuned1 = DecisionTreeClassifier(
    random_state=1, criterion="gini", max_depth=None, min_samples_split=2
)

# Fit the model on training data
dtree_tuned1.fit(X_train, y_train)


# In[298]:


# Calculating different metrics on train set
dtree_grid_train = model_performance_classification_sklearn(
    dtree_tuned1, X_train, y_train
)
print("Training performance:")
dtree_grid_train


# In[299]:


# Calculating different metrics on validation set
dtree_grid_val = model_performance_classification_sklearn(dtree_tuned1, X_val, y_val)
print("Validation performance:")
dtree_grid_val


# In[300]:


# creating confusion matrix
confusion_matrix_sklearn(dtree_tuned1, X_val, y_val)


# ### RandomizedSearchCV

# In[301]:


# Creating pipeline
model = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomizedSearchCV
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [2, 4, 7, 10, 15],
}
# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,
    scoring=scorer,
    cv=5,
    random_state=1,
)

# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train, y_train)

print(
    "Best parameters are {} with CV score={}:".format(
        randomized_cv.best_params_, randomized_cv.best_score_
    )
)


# In[302]:


# Creating new pipeline with best parameters
dtree_tuned2 = DecisionTreeClassifier(
    random_state=1, criterion="entropy", max_depth=None, min_samples_split=2
)

# Fit the model on training data
dtree_tuned2.fit(X_train, y_train)


# In[303]:


# Calculating different metrics on train set
dtree_random_train = model_performance_classification_sklearn(
    dtree_tuned2, X_train, y_train
)
print("Training performance:")
dtree_random_train


# In[304]:


# Calculating different metrics on validation set
dtree_random_val = model_performance_classification_sklearn(dtree_tuned2, X_val, y_val)
print("Validation performance:")
dtree_random_val


# In[305]:


# creating confusion matrix
confusion_matrix_sklearn(dtree_tuned1, X_val, y_val)


# ### XGBoost
# #### GridSearchCV

# In[306]:


get_ipython().run_cell_magic('time', '', '\n#defining model\nmodel = XGBClassifier(random_state=1,eval_metric=\'logloss\')\n\n#Parameter grid to pass in GridSearchCV\nparam_grid={\'n_estimators\':np.arange(50,150,50),\n            \'scale_pos_weight\':[2,5,10],\n            \'learning_rate\':[0.01,0.1,0.2,0.05],\n            \'gamma\':[0,1,3,5],\n            \'subsample\':[0.8,0.9,1],\n            \'max_depth\':np.arange(1,5,1),\n            \'reg_lambda\':[5,10]}\n\n\n# Type of scoring used to compare parameter combinations\nscorer = metrics.make_scorer(metrics.recall_score)\n\n#Calling GridSearchCV\ngrid_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs = -1, verbose= 2)\n\n#Fitting parameters in GridSeachCV\ngrid_cv.fit(X_train,y_train)\n\n\nprint("Best parameters are {} with CV score={}:" .format(grid_cv.best_params_,grid_cv.best_score_))\n')


# In[307]:


# building model with best parameters
xgb_tuned1 = XGBClassifier(
    random_state=1,
    n_estimators=50,
    scale_pos_weight=10,
    subsample=0.8,
    learning_rate=0.01,
    gamma=0,
    eval_metric="logloss",
    reg_lambda=5,
    max_depth=1,
)

# Fit the model on training data
xgb_tuned1.fit(X_train, y_train)


# In[308]:


# Calculating different metrics on train set
xgboost_grid_train = model_performance_classification_sklearn(
    xgb_tuned1, X_train, y_train
)
print("Training performance:")
xgboost_grid_train


# In[309]:


# Calculating different metrics on validation set
xgboost_grid_val = model_performance_classification_sklearn(xgb_tuned1, X_val, y_val)
print("Validation performance:")
xgboost_grid_val


# In[310]:


# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned1, X_val, y_val)


# #### RandomizedSearchCV

# In[311]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nmodel = XGBClassifier(random_state=1,eval_metric=\'logloss\')\n\n# Parameter grid to pass in RandomizedSearchCV\nparam_grid={\'n_estimators\':np.arange(50,150,50),\n            \'scale_pos_weight\':[2,5,10],\n            \'learning_rate\':[0.01,0.1,0.2,0.05],\n            \'gamma\':[0,1,3,5],\n            \'subsample\':[0.8,0.9,1],\n            \'max_depth\':np.arange(1,5,1),\n            \'reg_lambda\':[5,10]}\n\n# Type of scoring used to compare parameter combinations\nscorer = metrics.make_scorer(metrics.recall_score)\n\n#Calling RandomizedSearchCV\nxgb_tuned2 = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, scoring=scorer, cv=5, random_state=1, n_jobs = -1)\n\n#Fitting parameters in RandomizedSearchCV\nxgb_tuned2.fit(X_train,y_train)\n\nprint("Best parameters are {} with CV score={}:" .format(xgb_tuned2.best_params_,xgb_tuned2.best_score_))\n')


# In[312]:


# building model with best parameters
xgb_tuned2 = XGBClassifier(
    random_state=1,
    n_estimators=50,
    scale_pos_weight=10,
    gamma=1,
    subsample=0.9,
    learning_rate=0.01,
    eval_metric="logloss",
    max_depth=1,
    reg_lambda=5,
)
# Fit the model on training data
xgb_tuned2.fit(X_train, y_train)


# In[313]:


xgboost_random_train = model_performance_classification_sklearn(
    xgb_tuned2, X_train, y_train
)
print("Training performance:")
xgboost_random_train


# In[314]:


# Calculating different metrics on validation set
xgboost_random_val = model_performance_classification_sklearn(xgb_tuned2, X_val, y_val)
print("Validation performance:")
xgboost_random_val


# In[315]:


confusion_matrix_sklearn(xgb_tuned2, X_val, y_val)


# In[316]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        dtree_grid_train.T,
        dtree_random_train.T,
        xgboost_grid_train.T,
        xgboost_random_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree Tuned with Grid search",
    "Decision Tree Tuned with Random search",
    "Xgboost Tuned with Grid search",
    "Xgboost Tuned with Random Search",
]
print("Training performance comparison:")
models_train_comp_df


# In[317]:


# Validation performance comparison

models_val_comp_df = pd.concat(
    [
        dtree_grid_val.T,
        dtree_random_val.T,
        xgboost_grid_val.T,
        xgboost_random_val.T,
    ],
    axis=1,
)
models_val_comp_df.columns = [
    "Decision Tree Tuned with Grid search",
    "Decision Tree Tuned with Random search",
    "Xgboost Tuned with Grid search",
    "Xgboost Tuned with Random Search",
]
print("Validation performance comparison:")
models_val_comp_df


# In[318]:


feature_names = X_train.columns
importances = xgb_tuned1.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Pipeline for the Final Model 

# In[324]:


# Define numerical and categorical features
numerical_features = [
    "months_loan_duration",
    "amount",
    "percent_of_income",
    "years_at_residence",
    "age",
    "existing_loans_count",
    "dependents",
]

categorical_features = [
    "checking_balance",
    "credit_history",
    "purpose",
    "savings_balance",
    "employment_duration",
    "other_credit",
    "housing",
    "job",
    "phone",
]

# Preprocessing pipeline for numerical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Append classifier to preprocessing pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('default', axis=1), data['default'], test_size=0.2, random_state=1)


# In[325]:


# Creating new pipeline with best parameters
Final_pipeline_model = Pipeline(
    steps=[
        ("pre", preprocessor),
        (
            "XGB",
            XGBClassifier(
                random_state=1,
                n_estimators=50,
                scale_pos_weight=10,
                subsample=0.8,
                learning_rate=0.01,
                gamma=0,
                eval_metric="logloss",
                reg_lambda=5,
                max_depth=1,
            ),
        ),
    ]
)


# In[326]:


from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Fit the model on training data
Final_pipeline_model.fit(X_train, y_train_encoded)


# In[327]:


# Fit the model on training data
Final_pipeline_model.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred_encoded = Final_pipeline_model.predict(X_test)

# Convert the encoded predictions back to the original labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Encode the true labels of the test set
y_test_encoded = label_encoder.transform(y_test)

# Calculate different metrics on the test set
Test_Score = model_performance_classification_sklearn(Final_pipeline_model, X_test, y_test_encoded)
print("Final Model performance on Test Dataset:")
Test_Score


# ## Business Insights and Conclusions

# - In our analysis, we identified several crucial features that significantly impact credit default prediction: **months loan duration**, **checking balance**, and **credit history**. Understanding these factors is essential for informed decision-making in credit assessment.
# 
# - One notable observation is the model's high test recall of approximately 100%, indicating its effectiveness in identifying defaulters. However, this success is counterbalanced by a considerably low test precision of around 30%, revealing a weakness in correctly identifying non-defaulters. Consequently, the bank may miss out on opportunities to **extend credit to reliable customers**.
# 
# - To address this issue and enhance overall model performance, particularly in terms of precision, further refinements and adjustments are necessary. Once the desired level of model performance is attained, the bank can confidently employ the model for assessing creditworthiness among new customers.
# 
# - Our analysis also unveiled intriguing insights into customer behavior. We found that individuals with **limited or moderate savings in their checking accounts exhibit a higher propensity to default**. To mitigate this risk, the bank may consider implementing **stricter lending** criteria or adjusting interest rates accordingly.
# 
# - Furthermore, customers with **larger credit amounts** or **longer loan durations** demonstrate an increased likelihood of default. In response, the bank should exercise caution when extending substantial credit amounts or granting loans for extended periods.
# 
# - Moreover, our findings indicate that customers residing in **rented** or **free accommodations** are more susceptible to default. To proactively manage this risk, the bank should gather additional information, such as **hometown addresses**, to facilitate better tracking and risk assessment.
# 
# - Lastly, our analysis revealed a slight inclination towards default among **younger customers**. In light of this observation, the bank could revisit its policies to address this demographic trend and implement measures to mitigate associated risks.
# 
# - By incorporating these insights into their decision-making processes, the bank can bolster its credit assessment framework and make more informed lending decisions, ultimately safeguarding its financial interests while fostering responsible lending practices.
