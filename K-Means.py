#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[35]:


get_ipython().system('pip install scikit-learn')

get_ipython().system('pip install pandas')

get_ipython().system('pip install matplotlib')


# In[6]:


df = pd.read_csv("Mall_Customers.csv")


# In[7]:


df.head()


# In[8]:


plt.scatter(df["Age"],
           df["Spending_Score"])
plt.xlabel("Age")
plt.ylabel("Spending_Score")


# In[9]:


plt.scatter(df["Age"],
           df["Annual_Income_(k$)"])
plt.xlabel("Age")
plt.ylabel("Annual_Income_(k$)")


# In[13]:


plt.scatter(df["Spending_Score"], 
            df["Annual_Income_(k$)"])

plt.xlabel("Spending_Score")
plt.ylabel("Annual_Income_(k$)")


# In[16]:


df.isnull().sum()


# In[17]:


df_ = ["Age", "Annual_Income_(k$)" , "Spending_Score"]
df = df[df_]


# In[19]:


df.head()


# In[20]:


#Data Transformation
from sklearn.preprocessing import StandardScaler


# In[21]:


scaler = StandardScaler()
scaler.fit(df)
scalerdata = scaler.transform(df)


# In[26]:


scalerdata


# In[28]:


#Elbow method
def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
        
    
    return clusters_centers, k_values


# In[29]:


def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()


# In[36]:


clusters_centers, k_values = find_best_clusters(df, 12)

generate_elbow_plot(clusters_centers, k_values)


# In[40]:


kmeans_model = KMeans(n_clusters = 5)

kmeans_model.fit(df)


# In[41]:


df["clusters"] = kmeans_model.labels_

df.head()


# In[43]:


plt.scatter(df["Spending_Score"], 
            df["Annual_Income_(k$)"], 
            c = df["clusters"])


# In[ ]:




