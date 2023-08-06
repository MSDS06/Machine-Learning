#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

#warning

data = pd.read_csv("cancer.csv")
data.head(10)


# In[55]:


data.drop(["Unnamed: 32" , "id"] , inplace=True, axis=1)


# In[56]:


data.head(10)


# In[57]:


data = data.rename(columns={"diagnosis" : "target"})


# In[58]:


sns.countplot(data["target"])


# In[59]:


data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]


# In[60]:


data.head(10)


# In[61]:


data.shape


# In[62]:


data.info


# In[63]:


desciribe = data.describe()


# In[64]:


desciribe


# In[65]:



#standardization
#missing value: none
#EDA


# In[66]:


corr_matrix = data.corr()
print(corr_matrix)


# In[67]:


sns.clustermap(corr_matrix, annot = True , fmt= ".2f")
plt.title("Correlation Between Features")
plt.show()


# In[68]:


threshold = 0.5
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt=".2f")
plt.title("Correlation Between Features W Correlation Threshold 0.75")


# In[69]:


data_melted = pd.melt(data, id_vars="target",
                     var_name="features",
                     value_name="value")
plt.figure()
sns.boxplot(x="features" , y="value", hue="target", data=data_melted)
plt.xticks(rotation=90)
plt.show()


# In[70]:


sns.pairplot(data[corr_features], diag_kind = "kde", markers="+", hue="target")
plt.show()


# In[71]:


#skewness


# In[72]:


#outlier


# In[73]:


y = data.target
x = data.drop(["target"],axis = 1)
columns = x.columns.tolist()


# In[74]:


clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score


# In[ ]:





# In[75]:


threshold = -2.5
filter = outlier_score["score"] < threshold
outlier_index = outlier_score[filter].index.tolist()


# In[76]:


plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color="blue", s=50, label="Outlier")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="k", s=3, label="DataPoints")


# In[77]:


plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color="blue", s=50, label="Outlier")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="k", s=3, label="DataPoints")
radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s=1000*radius,edgecolors = "r" , facecolors="none" , label ="Outlier Scores")
plt.legend()
plt.show()


# In[78]:


x = x.drop(outlier_index)
y= y.drop(outlier_index).values


# In[79]:


type(y)


# In[80]:


type(x)


# In[81]:


#train test split


# In[82]:


test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42)


# In[83]:


X_test.shape


# In[84]:


X_train.shape


# In[85]:


#Standardization


# In[86]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns=columns)


# In[87]:


X_train_df_describe = X_train_df.describe()


# In[88]:


X_train_df_describe


# In[89]:


X_train_df["target"] = Y_train
data_melted = pd.melt(X_train_df, id_vars="target",
                     var_name="features",
                     value_name="value")
plt.figure()
sns.boxplot(x="features" , y="value", hue="target", data=data_melted)
plt.xticks(rotation=90)
plt.show()


# In[90]:


sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers="+", hue="target")
plt.show()


# In[91]:


#basic KNN method


# In[92]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc=accuracy_score(Y_test, y_pred)
score = knn.score(X_test, Y_test)
print("Score :" ,score)
print("CM :" ,cm)
print("Basic KNN Acc :" ,acc)


# In[93]:


#KNN best parameters


# In[94]:


def KNN_Best_Params(x_train, x_test, y_train, y_test):
    k_range = list(range(1,31))
    weight_options = ["uniform" , "distance"] 
    print()
    param_grid = dict(n_neighbors = k_range , weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring ="accuracy")
    grid.fit(x_train, y_train)
    
    print("Best Training Score: {} with parameters : {}" .format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test , y_pred_test)
    cm_train = confusion_matrix(y_train , y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score : {} , Train Score : {}" .format(acc_test, acc_train))
    print()
    print("CM Test : ", cm_test)
    print("CM Train : ", cm_train)
    
    return grid

grid = KNN_Best_Params(X_train , X_test , Y_train, Y_test)


# In[95]:


#PCA


# In[96]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns =["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1" , y = "p2", hue = "target", data = pca_data)
plt.title("PCA : p1 vs p2")


# In[97]:


X_train_pca , X_test_pca , Y_train_pca, Y_test_pca =train_test_split(X_reduced_pca, y, test_size = test_size, random_state=42)
grid_pca = KNN_Best_Params(X_train_pca , X_test_pca , Y_train_pca, Y_test_pca)


# In[98]:


#NCA


# In[105]:


nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state=42)
nca.fit(x_scaled , y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns=["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x="p1" , y="p2" , hue="target" ,data = nca_data)
plt.title("NCA : p1 vs p2")


# In[106]:


X_train_nca , X_test_nca , Y_train_nca, Y_test_nca =train_test_split(X_reduced_nca, y, test_size = test_size, random_state=42)
grid_nca = KNN_Best_Params(X_train_nca , X_test_nca , Y_train_nca, Y_test_nca)


# In[ ]:




