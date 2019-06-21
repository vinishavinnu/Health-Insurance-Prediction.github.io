
# coding: utf-8

# In[1]:


#step1:importing libraries


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#step2:importing dataset


# In[4]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_477537f25a89456a948518b4b3343fd4 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='nABSfFtOMeHoXpMFrmfJDW0-Z7AlO2yDSOml84_y9hoo',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_477537f25a89456a948518b4b3343fd4.get_object(Bucket='healthinsuranceprediction-donotdelete-pr-4m1f1uq3xalqjh',Key='insurance.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[5]:


type(dataset)


# In[23]:


x[1,:]


# In[6]:


dataset.isnull().any()


# In[7]:


x=dataset.iloc[:,0:6]
x


# In[8]:


x=dataset.iloc[:,0:6].values


# In[9]:


x


# In[10]:


y=dataset.iloc[:,-1:]
y


# In[11]:


y=dataset.iloc[:,-1:].values
y


# In[12]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[13]:


x[:,1]=lb.fit_transform(x[:,1])
x


# In[14]:


x[:,4]=lb.fit_transform(x[:,4])
x


# In[15]:


x[:,5]=lb.fit_transform(x[:,5])
x


# In[16]:


from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(categorical_features=[5])
x=oh.fit_transform(x).toarray()


# In[17]:


x


# In[18]:


x=x[:,1:]
x


# In[19]:


x.shape


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[22]:


x_train[0:1,0:]


# In[23]:


plt.scatter(x_train[:,5],y_train)


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


mr=LinearRegression()


# In[26]:


mr.fit(x_train,y_train)


# In[27]:


y_predict=mr.predict(x_test)
y_predict


# In[28]:


y_test


# In[29]:


mr.predict([[0. ,  0. ,  1. , 19. ,  0. , 27.9,  0. ,  1.]])


# In[30]:


from sklearn.metrics import r2_score


# In[31]:


s1=r2_score(y_test,y_predict)
s1


# # Decision tree

# In[32]:


from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)


# In[33]:


regressor.predict([[0. ,  0. ,  1. , 19. ,  0. , 27.9,  0. ,  1.]])


# In[34]:


y_predict =regressor.predict(x_test)


# In[35]:


y_predict


# In[36]:


regressor.predict([[0.   ,  0.   ,  1.   , 19.   ,  0.   , 27.9  ,  0.   ,  1.]])


# In[37]:


from sklearn.metrics import r2_score
s2=r2_score(y_test,y_predict)
s2


# # Random FOREST

# In[24]:


from sklearn.ensemble import RandomForestRegressor
RFclassifier=RandomForestRegressor(n_estimators=40,random_state=0)


# In[25]:


RFclassifier.fit(x_train,y_train)


# In[26]:


RFclassifier.predict([[0. ,  0. ,  1. , 19. ,  0. , 27.9,  0. ,  1.]])


# In[27]:


y_predict=RFclassifier.predict(x_test)
y_predict


# In[28]:


from sklearn.metrics import r2_score


# In[29]:


s3=r2_score(y_test,y_predict)


# In[30]:


s3


# # BAR GRAPH

# In[45]:


import matplotlib.pyplot as plt


# In[46]:


x1=["MultiLinear","DecisionTree","RandomForest"]
y1=[s1,s2,s3]
plt.bar(x1,y1,label="Algorithm score")
plt.xlabel("Regression Algorithms")
plt.ylabel("r2_score values")
plt.legend()
plt.show()


# In[47]:


get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# In[48]:


sns.kdeplot(dataset[dataset['sex']=='female']['charges'], shade=True, label = 'Female charge')
sns.kdeplot(dataset[dataset['sex']=='male']['charges'], shade=True, label = 'Male charge')


# In[49]:


sns.swarmplot(x='sex', y='charges', data=dataset)


# In[50]:


#The impact of smoke on charges

dataset.groupby("smoker")['charges'].agg('mean').plot.bar()


# In[52]:


sns.regplot(x='bmi',y='charges',data=dataset)


# In[53]:


sns.lmplot(x='bmi',y='charges',hue='sex',data=dataset)


# In[56]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[31]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[32]:


wml_credentials={"access_key": "geAmgw9k3xfVQUURU7MxMu_F4cR9D60rsgm1hjyVGjUe"
    ,"instance_id": "8e3540a3-21f4-4961-add6-24ce334b12e0",
  "password": "a41fa7d7-36d7-4b2f-b700-12e71e15f7c1",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "2e458276-f253-48a1-927b-e5c2ee769b7d"
                }


# In[33]:


client = WatsonMachineLearningAPIClient(wml_credentials)
import json


# In[34]:


instance_details = client.service_instance.get_details()
print(json.dumps(instance_details, indent=2))


# In[35]:


model_props = {client.repository.ModelMetaNames.AUTHOR_NAME: "Vinisha", 
               client.repository.ModelMetaNames.AUTHOR_EMAIL: "vinishavinnu10@gmail.com", 
               client.repository.ModelMetaNames.NAME: "Regression"}


# In[36]:


model_artifact =client.repository.store_model(RFclassifier, meta_props=model_props)


# In[37]:


published_model_uid = client.repository.get_model_uid(model_artifact)


# In[38]:


published_model_uid


# In[39]:


created_deployment = client.deployments.create(published_model_uid, name="Regression")


# In[40]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployment)
scoring_endpoint


# In[41]:


client.deployments.list()

