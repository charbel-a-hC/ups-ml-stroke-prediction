#!/usr/bin/env python
# coding: utf-8

# ## Importing Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
import sklearn

# data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# metrics
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, accuracy_score, 
                             f1_score, mean_squared_error, roc_curve, auc, balanced_accuracy_score)
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, RocCurveDisplay, classification_report

# model selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# oversampling
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE

# neural network
from sklearn.neural_network import MLPClassifier
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ## Data Analysis

# In[2]:


dataframe = pd.read_csv("healthcare-dataset-stroke-data.csv")
dataframe


# In[3]:


# data shape
rows, cols = dataframe.shape
print(f"The dataset is composed of {rows} rows and {cols} columns.")


# In[4]:


dataframe.info()


# We observe multiple NaN values for the <*bmi*> category (exactly 201 values) which make up approximately **4%** of the data for this feature. We can look into replacing the NaN values with the mean for this category.
# We also drop the **id** comlumn from our dataset as it has no influence on our expected outcome.

# In[5]:


dataframe = dataframe.drop(labels= "id", axis=1)
dataframe.head()


# ### Some Statistics For the Numerical Data

# In[6]:


dataframe.describe()


# ### Representing Unique Values of Categorical Features

# In[7]:


# get categorical data having multiple labels
categorical = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
categorical_df = pd.DataFrame()
for categorical_feature in categorical:
    tmp_df = pd.DataFrame({categorical_feature:dataframe[categorical_feature].unique()})
    categorical_df = pd.concat([categorical_df, tmp_df], axis=1)
categorical_df.head()


# ### Visualizing Distribution of Categorical Features

# In[8]:


X = dataframe.drop(labels=["stroke"], axis=1)
Y = dataframe["stroke"]

X.head()


# In[9]:


all_categories = categorical

fig, ax = plt.subplots(4, 2, figsize=(12,5), constrained_layout=True)
fig.delaxes(ax[3,1])
fig.set_figheight(12)
for i, cat_var in enumerate(all_categories):
    try:
        j,k = np.unravel_index(i, shape= (4, 2))
        cp = sns.countplot(y=cat_var, data=X, label='features', ax=ax[j,k])
        ax[j,k].set_title(cat_var)
    except Exception as e:
        print(e)
plt.show()


# ### Visualizing Distribution of Target

# #### Helper Function - Target Distribution Visualizer

# In[10]:


def vis_target_dist(target_var, title):
    
    pie_data = target_var.value_counts(normalize=True).values * 100
    pie_label = target_var.value_counts(normalize=True).index

    fig, ax = plt.subplots(figsize=(8,6))

    wedges, texts, autotexts = ax.pie(pie_data, startangle=0, explode=[0, 0.2],
                                      autopct='%.2f%%', textprops={'color':'w', 'fontsize':14, 'weight':'bold'})

    ax.legend(wedges, pie_label,
              title='Stroke',
              loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(title)
    plt.tight_layout()


# In[11]:


vis_target_dist(Y, "Target Variable Distribution")
plt.show()


# ### Finding Correlation of the Categorical Features
# In this part, we want to represent the correlation between each of the features in the dataset against the target value *stroke*. Pearson's correleation ([source](https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php)) can be easily obtained using the `pandas` library. This approach allows us to recognize the most relevant features in our dataset by identifying the ones with the lowest correlation value calculated by Pearson's correlation.

# In[12]:


# get a one-hot-encoded representation of the categorical data (required for the Pearson Correlation calculation)
dataframe_ohe = pd.get_dummies(data=dataframe)
dataframe_ohe


# In[13]:


# setting a figure size for the heatmap
plt.figure(figsize=(16,12))
# calculating correlation and storing it in a dataframe
cor = dataframe_ohe.corr()
cor.head()


# In[14]:


plt.figure(figsize=(16,12))
ax = sns.heatmap(cor.round(2), square=True, cmap=plt.cm.Reds, annot=True)
for i in dataframe_ohe.index[dataframe_ohe['stroke'] == True].tolist():
    j = dataframe_ohe.columns.get_loc('stroke')
    ax.add_patch(Rectangle((i, j), 1, 1, ec='black', fc='none', lw=2, alpha=0.7))
    ax.add_patch(Rectangle((j, i), 1, 1, ec='black', fc='none', lw=2, alpha=0.7))
plt.show()


# In[15]:


#Correlation with output variable
cor_target = abs(cor["stroke"])
#Selecting highly correlated features
cor_target


# From the correlation values calculated, for each output in a categorical feature, if one output isn't highly correlated to the target **stroke**, we can deduce that the entire feature is relevant and not correlated.
# All of the features in our dataset are relevant. We note the 3 most relevant based on correlation values:
# - **gender**
# - **work_type**
# - **smoking_status/Residence_type**

# ### Inspecting Missing or NaN Values

# In[16]:


def summarize_missingness(df):
    '''
    Utility function to summarize missing or NaN values
    '''
    nulls = df.isnull()
    counts = nulls.sum()
    percs = nulls.mean().mul(100.)
    
    nulls_df = pd.DataFrame({'Count of missing/NaN values': counts, 'Percentage of missing values': percs}, 
                            index=counts.index)
    
    display(nulls_df)


# In[17]:


vars_with_na = [col for col in dataframe_ohe if dataframe_ohe[col].isnull().sum() > 0]
summarize_missingness(dataframe[vars_with_na])


# We can see that only the feature **bmi** has 201 missing or NaN values which represent ~4% of the datapoints.

# ### Some More EDA
# In this part, we inspect the features of the dataset in function of the target class **stroke**. We seek to further understand the underlying relation between the dataset features and the output obtained.

# #### Numerical Variables Vs Stroke
# The numerical variables in our dataset are **age**, **bmi** and **avg_glucose_level**. In this part, we will try to comprehend the effect of each of these variables on our target.

# In[18]:


num_vars = dataframe.select_dtypes(include=['float']).columns.tolist()
num_vars


# In[19]:


target_var = "stroke"
color1, color2 = "blue", "green"
for num_var in num_vars:
    fig, ax = plt.subplots(nrows= 1, ncols= 2 )
    fig.set_figheight(5)
    fig.set_figwidth(9)

    ax[0].hist(dataframe[dataframe[target_var]==1][f"{num_var}"], bins=15, alpha=0.5, color=color1, label="had a stroke")
    ax[0].hist(dataframe[dataframe[target_var]==0][f"{num_var}"], bins=15, alpha=0.5, color=color2, label="did not have a stroke")

    ax[0].set_xlabel(num_var)
    ax[0].set_ylabel("Count of Patients")
    ax[0].legend();


    sns.kdeplot(dataframe[dataframe[target_var]==1][num_var], shade=True, color=color1, label="had a stroke", ax=ax[1])
    sns.kdeplot(dataframe[dataframe[target_var]==0][num_var], shade=True, color=color2, label="did not have a stroke", ax=ax[1])

    ax[1].set_xlabel(num_var)
    ax[1].set_ylabel("Density")
    ax[1].legend();
    fig.suptitle(f"{num_var} vs. {target_var} for Patients");
    plt.tight_layout()
    plt.show()


# #### Categorical Variables Vs Stroke

# In[20]:


cat_vars = categorical
cat_vars


# In[21]:


target_var = "stroke"
for cat_var in cat_vars:
    tmp_counts_df = dataframe.groupby([cat_var, target_var])["age"].count().unstack()
    
    tmp_target_perc_df = tmp_counts_df.T.div(tmp_counts_df.T.sum()).T
    tmp_feature_perc_df = tmp_counts_df.div(tmp_counts_df.sum()).T

    fig, ax = plt.subplots(nrows=1, ncols= 2)
    fig.set_figheight(5)
    fig.set_figwidth(9)
    
    tmp_target_perc_df.plot(kind="bar", stacked=True, color=["green", "blue"], ax=ax[0])
    ax[0].set_xlabel(cat_var)
    ax[0].set_ylabel("Proportion")
    color_patches = [
        Patch(facecolor="blue", label="had a stroke"),
        Patch(facecolor="green", label="did not have a stroke")
    ]
    ax[0].legend(handles=color_patches)
    
    tmp_feature_perc_df.plot(kind="bar", stacked=True, ax=ax[1])
    ax[1].legend(title=cat_var)
    ax[1].set_xticklabels(["did not have a stroke", "had a stroke"], rotation=0)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Proportion")
    
    fig.suptitle(f"{cat_var} vs. Stroke for Patients");


# #### Numerical Variables

# In[22]:


combs = [("age","bmi"), ("age", "avg_glucose_level"), ("bmi", "avg_glucose_level")]
target_var = "stroke"

for comb in combs:
    num_var1, num_var2 = comb
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(dataframe[dataframe[target_var]==1][num_var1], dataframe[dataframe[target_var]==1][num_var2], c="blue", alpha=0.5)
    ax.scatter(dataframe[dataframe[target_var]==0][num_var1], dataframe[dataframe[target_var]==0][num_var2], c="green", alpha=0.5)

    ax.set_xlabel(num_var1)
    ax.set_ylabel(num_var2)

    color_patches = [
        Line2D([0], [0], marker='o', color='w', label='had a stroke', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='did not have a stroke', markerfacecolor='g', markersize=10)
    ]
    ax.legend(handles=color_patches)

    fig.suptitle(f"Stroke by {num_var1} and {num_var2} for Patients");


# ## Feature Engineering
# After visualizing all of the features in our dataset, we notice some NaN values which we need to remove, we also need to normalize the numerical features as well as establish one-hot-encoded representations of the features. We might also be interested in removing the *Others* value for the gender as it represents 0.02% of the **gender** feature. 

# ### Gender Feature
# For this feature, we remove the *Others* value

# In[23]:


# get the index of the row having "Other" as value for gender
Other_idx = dataframe.index[dataframe['gender']=="Other"].tolist()
dataframe_cleaned = dataframe.drop(Other_idx)
dataframe_cleaned


# In[24]:


# representing the categorical values after dropping
# get categorical data having multiple labels
categorical = categorical

categorical_df = pd.DataFrame()
for categorical_feature in categorical:
    tmp_df = pd.DataFrame({categorical_feature:dataframe_cleaned[categorical_feature].unique()})
    categorical_df = pd.concat([categorical_df, tmp_df], axis=1)
categorical_df.fillna("", inplace= True)
categorical_df.head()


# ### BMI Feature
# We will fill the NaN values with the mean values for the BMI features as there are 240+ datapoints with NaN values we might risk loosing important information.

# In[25]:


dataframe_cleaned['bmi'].fillna((dataframe_cleaned['bmi'].mean()), inplace= True)

vars_with_na = [col for col in dataframe_cleaned if dataframe_cleaned[col].isnull().sum() > 0]
summarize_missingness(dataframe_cleaned[vars_with_na])


# ### Feature Scaling
# Feature scaling is the process of scaling numerical features either by min-max scaling or standardizations so that we keep our values in a defined range for each feature.
# - For standardization: $${x_{stand}}=\frac{{x_{orig}-\mu}}{\sigma}$$ where $\mu$ is the mean and $\sigma$ is the standard deviation.
# - min-max scalar: $${x_{norm}}=\frac{{x_{orig}-min}}{max-min}$$
# We will apply the min-max technique since our dataset distribution is not normal. As a result we will have values between 0 and 1.

# In[26]:


cat_vars = categorical + ["stroke"]
# intialize output dataframe
dataframe_scaled = pd.DataFrame()

# separate categorical and numerical features into two different dataframes
dataframe_cleaned_cat = dataframe_cleaned[[c for c in dataframe_cleaned.columns if c in cat_vars]]
dataframe_cleaned_num = dataframe_cleaned[[c for c in dataframe_cleaned.columns if c in num_vars]]

# scale numerical values using Standard scalar
scaler = MinMaxScaler()
dataframe_scaled_num = pd.DataFrame(scaler.fit_transform(dataframe_cleaned_num), columns= num_vars)

# reset index of categorical data points (index resets after scaling - difference in scaling comes from removing "Others")
# data points from the dataset
dataframe_cleaned_cat = pd.DataFrame(np.array(dataframe_cleaned_cat), columns = cat_vars)

# concatenating both dataframes
dataframe_scaled = pd.concat([dataframe_scaled_num, dataframe_cleaned_cat], axis= 1)

# converting binary categories to int64 since after passing to array it's transformed to Object
binary_cat = ["hypertension", "heart_disease", "stroke"]
dataframe_scaled[binary_cat] = dataframe_scaled[binary_cat].astype(np.int64)

dataframe_scaled


# ### One-Hot-Encoding Non-Binary Categorical Features

# In[27]:


dataframe_ohe_scaled = pd.get_dummies(data= dataframe_scaled)
dataframe_ohe_scaled


# ### Getting Feature Matrix and Target Vector

# In[28]:


X = dataframe_ohe_scaled.drop("stroke", axis= 1) #Feature Matrix
y = dataframe_ohe_scaled["stroke"] #Target Variable


# ### Using Correlation to Filter-Out Features
# From the calculated correlation matrix, we are able to filter-out low-correlation features with the target since they do not provide any information with regards to the target.

# In[29]:


corr_matrix = cor.round(3)
corr_target = corr_matrix['stroke'].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,6))

sns.barplot(x=corr_target.index, y=corr_target.values, ax=ax)

ax.grid(False)
ax.set_title('Attribute Correlation to Stroke')
plt.setp(ax.get_xticklabels(), rotation=90)

for n, x in enumerate(corr_target.index):
    if corr_target[n] >= 0:
        ax.text(x=n, y=corr_target[n], s=corr_target[n].round(2),
            horizontalalignment='center', verticalalignment='bottom',
            fontsize=14, fontweight='bold')
    else:
        ax.text(x=n, y=corr_target[n], s=corr_target[n].round(2),
            horizontalalignment='center', verticalalignment='top',
            fontsize=14, fontweight='semibold')

ax.axis('tight')

plt.show()


# We can select the following features having the highest correlation with respect to the target:
# - age
# - heart_disease
# - avg_glucose_level
# 
# We can also select some others having some lower correlations but provide information about the targets:
# - hypertension
# - ever_married (Yes/No)

# In[30]:


X_filtered = X[["age", "hypertension", "heart_disease", "avg_glucose_level", "ever_married_No", "ever_married_Yes"]]
y = dataframe_ohe_scaled["stroke"]
X_filtered


# It is important to note that we can also keep the remaining features but after some performed experiments, fitlering features increased the model performance as we removed noisy data which did not contribute to the target prediction.

# ## Model Training
# To train any machine learning model, we must follow the steps below: 
# 1. obtain a processed dataset with relevant features - split into train/val/test
# 2. define project/model metrics based on which we benchmark models - either accuracy, precision, recall, f1-score or ROC curve
# 3. select an appropriate model to best fit our data
# 4. train a model and evaluate performance based on validation data and metrics
# 5. test model on testing data
# 6. if we need some improvements thorugh hyperparameter tuning we repeat from step 3

# ### Splitting The Dataset

# In[31]:


# set seed
SEED = 100

# separate data into train and test
# test_size= 0.15 => 15% testing 85% training
# Fix the seed to the random generator
X_train, X_test, y_train, y_test = train_test_split(np.array(X_filtered), np.array(y), test_size=0.15, random_state=SEED)

print(f"Shape of training features {X_train.shape}")
print(f"Shape of test features {X_test.shape}")
print(f"Shape of training targets {y_train.shape}")
print(f"Shape of testing targets {y_test.shape}")


# In[32]:


vis_target_dist(pd.DataFrame(y_train), "y_train Distribution")
vis_target_dist(pd.DataFrame(y_test), "y_test Distribution")

plt.show()


# ### Metrics To Consider

# Confusion Matrix, Precision, Recall, f1-score, Precision-recall curve.

# In[33]:


def construct_classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    diag_cm = cm.diagonal()
    class_pos_acc = diag_cm[0]
    class_neg_acc = diag_cm[1]

    balanced_acc = (class_neg_acc + class_pos_acc)/2
    
    cr = classification_report(y_true, y_pred)
    aug_cr = cr[:53] + "\taccuracy" + cr[53:108] + f"\t    {class_pos_acc:.2f}" + cr[108:162]        + f"\t    {class_neg_acc:.2f}" + cr[162:217] + f"\tbalanced accuracy: {balanced_acc:.2f}" + cr[217:]
    
    return aug_cr


# In[34]:


def vis_metrics(y_true, y_pred, model_name= None):
    
    print(construct_classification_report(y_true, y_pred))
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels= [1, 0])
    disp.ax_.set_title("Confusion Matrix")

    
    # plot PR curve
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_pred, pos_label= 1, name= model_name)
    disp.ax_.set_title("P-R Curve")
    
    disp = RocCurveDisplay.from_predictions(y_true, y_pred, pos_label= 1, name= model_name)
    disp.ax_.set_title("ROC Curve")


# ### Model Training
# In this part, we will implement 3 models for classification:
# - Random Forest
# - Logistic Regression
# - Multi-Layered Perceptron (Neural Network)

# #### Logistic Regression Model

# In[35]:


logistic_model = LogisticRegression(max_iter= 1000)
logistic_model.fit(X_train, y_train)

# Predicting on the test data
pred_normal = logistic_model.predict(X_train)

vis_metrics(y_train, pred_normal, "LogisticModel")
plt.show()


# We note that the performance of this model is rather low with no regard to our positive class which is a patient having a stroke. This target is our main objective and low performance on that target should not be accepted for this model.
# This low performance comes from the fact that our target distribution from the training set is not balanced. Thus the learning process favors the overall accuracy but not one class accuracy. We will use the following methods to try and counter this issue:
# - **Assigning Class Weights**: it's the process of assigning weights to each of the target classes favoring one class over the other during training; the class with the higher weight will impose a larger penalty on the model.
# - **Over-sampling**: Oversampling is the process of randomly resampling the training dataset (and not the test set) in a way to favor the minority class and increase its representation in the training set. We will implement two methods: Random and ADASYN by using the *imbalanced-learn* library. 
# 
# These methods will be implemented for all 3 training models.

# #### Logistic Regression Model with Weighted Classes
# 
# ##### Obtaining Class Weights
# 
# Available scoring metrics to obtain the best class weights.

# In[36]:


def get_class_weights(X_train, y_train, scoring= 'f1'):
    lr = LogisticRegression(solver='newton-cg')

    #Setting the range for class weights
    weights = np.linspace(0.0,0.99,200)

    #Creating a dictionary grid for grid search
    param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

    #Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator= lr, 
                              param_grid= param_grid,
                              cv=StratifiedKFold(), 
                              n_jobs=-1, 
                              scoring= scoring, 
                              verbose=2).fit(X=X_train, y=y_train)

    #Ploting the score for different values of weight
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()

    weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})

    x, y = weigh_data['weight'], weigh_data['score']

    def annot_max(x,y, ax=None):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

        return xmax, ymax

    sns.lineplot(x, y)
    xmax, ymax = annot_max(x,y)
    plt.xlabel('Weight for class 1')
    plt.ylabel(scoring)
    plt.xticks([round(i/10,1) for i in range(0,11,1)])
    plt.title('Scoring for different class weights')
    plt.show()
    
    return xmax, ymax


# In[37]:


scoring = 'f1'
#sklearn.metrics.SCORERS.keys() Scoring techniques can be shown from this output


# In[38]:


xmax, ymax = get_class_weights(X_train, y_train, scoring= scoring)


# ##### Logistic Regression Model Trainning + Metrics Visualization

# In[39]:


logistic_model = LogisticRegression(solver='newton-cg', class_weight={0: (0.99-xmax), 1:xmax}, max_iter= 1000)
logistic_model.fit(X_train, y_train)

# Predicting on the test data
pred_weighted = logistic_model.predict(X_test)

vis_metrics(y_test, pred_weighted, "LogisticModelWeighted")
plt.show()


# #### Over-Sampling Dataset - Random

# In[40]:


ros = RandomOverSampler(random_state=0)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)


# In[41]:


vis_target_dist(pd.DataFrame(y_train_ros), "Random Over-Sampled Data")


# In[42]:


logistic_model = LogisticRegression(solver='newton-cg', max_iter= 1000)
logistic_model.fit(X_train_ros, y_train_ros)

# Predicting on the test data
pred_rand_overs = logistic_model.predict(X_test)

vis_metrics(y_test, pred_rand_overs, "LogisticModelRandomOverSamp")
plt.show()


# #### Over-Sampling Dataset - ADASYN

# In[43]:


X_train_adasyn, y_train_adasyn = ADASYN().fit_resample(X_train, y_train)


# In[44]:


vis_target_dist(pd.DataFrame(y_train_adasyn), "ADASYN Over-Sampled Data")


# In[45]:


logistic_model = LogisticRegression(solver='newton-cg', max_iter= 1000)
logistic_model.fit(X_train_adasyn, y_train_adasyn)

# Predicting on the test data
pred_adasyn = logistic_model.predict(X_test)

vis_metrics(y_test, pred_adasyn, "LogisticModelADASYN")
plt.show()


# We observe a good improvement over all the defined metrics when doing oversampling of the training dataset.
# Oversampling is shown to have a better performance compared to class weights with a considerable increase on the recall to 81% and an increase to the unbalanced accuracy to 76% from 70%.
# In this next part, we will use the over-sampled data on a neural network and on a random forset model since over-sampling proved to be the better technique in our case to combat the dataset distribution.

# #### Neural Network

# In[51]:


nn = MLPClassifier(hidden_layer_sizes= (10, 5, 1))
nn.fit(X_train_adasyn, y_train_adasyn)


# In[52]:


pred_nn = nn.predict(X_test)
vis_metrics(y_test , pred_nn, "MLP-ADASYN")


# #### Random Forest

# In[48]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train_adasyn, y_train_adasyn)

pred_rf = rf_model.predict(X_test)
vis_metrics(y_test, pred_rf, "RandomForest-ADASYN")
plt.show()


# ### Model Evaluation

# In this section, we evaluate all the three models implemented based on the defined metrics. 

# In[53]:


print("Classification report for LOGISTIC REGRESSION - ADASYN")
print(classification_report(y_test, pred_adasyn))

print("Classification report for NEURAL NETWORK - ADASYN")
print(classification_report(y_test, pred_nn))

print("Classification report for RANDOM FOREST - ADASYN")
print(classification_report(y_test, pred_rf))


# In[54]:


fpr_lr, tpr_lr, _ = roc_curve(y_true= y_test, y_score= pred_adasyn)
fpr_nn, tpr_nn, _ = roc_curve(y_true= y_test, y_score= pred_nn)
fpr_rf, tpr_rf, _ = roc_curve(y_true= y_test, y_score= pred_rf)

print(f"False Positive Rate/True Positive Rate for LOGISTIC REGRESSION - ADASYN: {fpr_lr[1]:.2f} - {tpr_lr[1]:.2f}")
print(f"False Positive Rate/True Positive Rate for NEURAL NETWORK - ADASYN: {fpr_nn[1]:.2f} - {tpr_nn[1]:.2f}")
print(f"False Positive Rate/True Positive Rate for RANDOM FOREST - ADASYN: {fpr_rf[1]:.2f} - {tpr_rf[1]:.2f}")


# From the following data and the plots shown previously, we could possibly deploy the neural network model having the highest true positives rate. 
# There are some other techniques we could implement to have a better assessment for our model however, due to the time limitations for this study we will suffise with the performed study.
