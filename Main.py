
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import json


# In[3]:


with open("yelp_academic_dataset_review.json", encoding="utf8") as f:
    data = f.readlines()


# In[4]:


jsondat = []
for f in data:
    jsondat.append(json.loads(f))


# In[5]:


userset = set()
restset = set()
for data in jsondat:
    userset.add(data['user_id'])
    restset.add(data['business_id'])


# In[6]:


mapuser={}
i = 1
mapbackuser = {}
for data in userset:
    mapuser[data] = i
    mapbackuser[i] = data
    i+=1
i=1
maprest = {}
mapbackrest = {}
for data in restset:
    maprest[data] = i
    mapbackrest[i] = data
    i+=1


# In[55]:


actualdata=[]


# In[56]:


print (len(jsondat))


# In[57]:


import csv


# In[58]:


with open('train.dat', "w", encoding = "latin-1") as f:
    writer = csv.writer(f, delimiter = ",")
    i=0
    thresh = len(jsondat)*0.8
    for data in jsondat:
        if (mapuser[data['user_id']]<=100000 and maprest[data['business_id']]<=10000):
            actualdata.append([str(mapuser[data['user_id']]), str(maprest[data['business_id']]), str(data['stars']), str(99999)])
#             writer.writerow([str(mapuser[data['user_id']]), str(maprest[data['business_id']]), str(data['stars'])])
#         if (i>thresh):
#             break
# with open('test.dat', "w", encoding = "latin-1") as f:
#     writer = csv.writer(f, delimiter = ",")
#     i=0
#     thresh = len(jsondat)*0.8
#     for data in jsondat:
#         i+=1
#         if (maprest[data['business_id']]>=):
#             writer.writerow([str(mapuser[data['user_id']]), str(maprest[data['business_id']]), str(data['stars'])])
 


# In[43]:


with open('train.dat', "w", encoding = "latin-1") as f:
    writer = csv.writer(f, delimiter = ",")
    i=0
    thresh = len(actualdata)*0.8
    for data in actualdata:
        i+=1
        if (i<thresh):
#             actualdata.append([str(mapuser[data['user_id']]), str(maprest[data['business_id']]), str(data['stars'])])
            writer.writerow(data)


# In[44]:


with open('test.dat', "w", encoding = "latin-1") as f:
    writer = csv.writer(f, delimiter = ",")
    i=0
    thresh = len(actualdata)*0.8
    
    for data in actualdata:
        i+=1
        if (i>=thresh):
#             actualdata.append([str(mapuser[data['user_id']]), str(maprest[data['business_id']]), str(data['stars'])])
            writer.writerow(data)


# In[59]:


actualdata = [list(map(int,i)) for i in actualdata]
actualdata = np.array(actualdata)


# In[60]:


np.shape(actualdata)


# In[62]:


actualdata[:,0]
test = actualdata[(int)(len(actualdata)*0.8):]
actualdata = actualdata[:(int)(len(actualdata)*0.8)]


# In[48]:


import pandas as pd
import json
import numpy as np
import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Model
from keras.layers import dot
import warnings
warnings.filterwarnings('ignore')


# In[49]:


n_movies = 9999
n_users = 99999
n_latent_factors = 5


# In[50]:



movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input))

prod = dot([movie_vec, user_vec], axes = 1)

model = Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error')
model.summary()


# In[51]:


history = model.fit([actualdata[:,0], actualdata[:,1]], actualdata[:,2], epochs=100, verbose=0)


# In[53]:


from sklearn.metrics import mean_squared_error
from math import sqrt

y_hat = np.round(model.predict([test[:,0], test[:,1]]),0)
y_true = test[:,2]
sqrt(mean_squared_error(y_true, y_hat))


# In[21]:


len(mapuser)


# In[22]:


len(maprest)


# In[63]:


len(test)

