#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import seaborn as sns


# In[4]:


sns.set_style("darkgrid")


# In[12]:


df=pd.read_csv(f"C:\\Users\\egeme\\Downloads\\starbucks.csv",index_col=0)


# In[13]:


df.head()


# In[14]:


df.info()


# In[15]:


df.describe()


# In[19]:


df["item"].unique()


# In[20]:


df["type"].unique()


# In[23]:


df.groupby("type")["item"].count()


# In[24]:


df.groupby("type")["item"].count().plot()
plt.title("Type of Items")
plt.text(0.,50, "Product Category Range",bbox=dict(facecolor="yellow",alpha=0.5))
plt.show()


# In[25]:


sns.countplot(x="type",data=df,palette="Set1")
plt.title("Product count")
plt.show()


# In[26]:


sns.catplot(kind="bar",x="type",y="calories",data=df)
plt.title("Calories count")
plt.show()


# In[27]:


sns.catplot(kind="bar",x="type",y="protein",data=df)
plt.title("Calories count")
plt.show()


# In[28]:


sns.catplot(kind="bar",x="type",y="carb",data=df)
plt.title("Calories count")
plt.show()


# In[29]:


sns.catplot(kind="bar",x="type",y="fiber",data=df)
plt.title("Calories count")
plt.show()


# In[32]:


numeric_columns = df.select_dtypes(include='number')
corr_matrix = numeric_columns.corr()


# In[34]:


df.corr(numeric_only=True)


# In[35]:


sns.heatmap(df.corr(numeric_only=True),annot=True)


# In[36]:


plt.title("Calorie and Fat Correlation")
sns.scatterplot(x="calories",y="fat",data=df,s=30,edgecolor="red")
plt.plot()


# In[38]:


sns.displot(x="calories",data=df,color="red",kde=True)
plt.title("Calorie Graph")
plt.show()


# In[40]:


sns.displot(x="protein",data=df,color="green",kde=True)
plt.title("Protein Graph")
plt.show()


# In[41]:


sns.displot(x="fat",data=df,color="purple",kde=True)
plt.title("Fat Graph")
plt.show()


# In[43]:


sns.displot(x="carb",data=df,color="violet",kde=True)
plt.title("Carb Graph")
plt.show()


# In[44]:


df.head()


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[47]:


x=df[["calories","fat","carb","fiber","protein"]]
y=df["type"]


# In[48]:


x_train,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[50]:


model=DecisionTreeClassifier()
model.fit(x_train,ytrain)


# In[51]:


y_pred=model.predict(xtest)


# In[52]:


accuracy=accuracy_score(ytest,y_pred)


# In[54]:


print("Truthfulness: ",accuracy)


# In[59]:


prediction=model.predict([[300,3,60,1,5]])
print(prediction)


# In[64]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# In[76]:


feature_name = list(x.columns)
class_name = list(model.classes_)
plt.figure(figsize=(15,10))
plot_tree(model,feature_names=feature_name,class_names=class_name,filled=True)
plt.show()


# In[ ]:




