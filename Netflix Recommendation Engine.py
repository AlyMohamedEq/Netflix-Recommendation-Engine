#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


netflix = pd.read_csv(r'E:\Basics of ML & Python\Netflix_titles.csv')


# In[3]:


netflix.head()


# In[4]:


netflix.info()


# In[5]:


netflix.isna().sum()


# In[6]:


netflix.dropna(inplace=True)


# In[7]:


netflix.isna().sum()


# In[8]:


netflix.dtypes


# In[9]:


netflix1 = netflix[['type','title','country','release_year','rating','listed_in','description']]


# In[10]:


netflix1.head()


# In[11]:


movie = netflix1[netflix1['type'] == 'Movie']


# In[12]:


movie.head()


# In[13]:


TVshow = netflix1[netflix1['type'] == 'TV Show']


# In[14]:


TVshow.head()


# In[15]:


TVshow.shape


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[17]:


Tfidfvec = TfidfVectorizer(min_df= 2, max_df= 0.7)


# In[18]:


Tvshow_vectorized = Tfidfvec.fit_transform(TVshow['description'])


# In[19]:


print(Tfidfvec.get_feature_names())  #method requires()#


# In[20]:


print(Tvshow_vectorized.toarray())


# In[21]:


TVshow_vector = pd.DataFrame(Tvshow_vectorized.toarray(), columns = Tfidfvec.get_feature_names(), index=TVshow['title'])


# In[22]:


print(TVshow_vector)


# In[23]:


from sklearn.metrics.pairwise import cosine_similarity


# In[24]:


Final_TVshow = cosine_similarity(TVshow_vector)


# In[25]:


print(Final_TVshow)


# In[26]:


TVshow_df = pd.DataFrame(Final_TVshow, index=TVshow['title'], columns=TVshow['title'])


# In[27]:


TVshow_df


# In[28]:


#recommendations for TVshows
TVshow_recommender = TVshow_df.loc['46'].sort_values(ascending = False)


# In[29]:


print(TVshow_recommender.head(11))


# In[30]:


movie.shape


# In[31]:


TfidfvecMovie = TfidfVectorizer(min_df= 2, max_df= 0.7)


# In[32]:


Movie_vectorized = TfidfvecMovie.fit_transform(movie['description'])


# In[33]:


print(TfidfvecMovie.get_feature_names())


# In[34]:


print(Movie_vectorized.toarray())


# In[35]:


movie_vector = pd.DataFrame(Movie_vectorized.toarray(),index=movie['title'], columns=TfidfvecMovie.get_feature_names())


# In[36]:


print(movie_vector)


# In[37]:


Final_movie = cosine_similarity(movie_vector)


# In[38]:


print(Final_movie)


# In[39]:


Movie_df = pd.DataFrame(Final_movie, index=movie['title'], columns=movie['title'])


# In[40]:


Movie_df


# In[41]:


#recommendations for Movies

recommended_movies = Movie_df.loc['Layer Cake'].sort_values(ascending = False)


# In[42]:


print(recommended_movies.head(11))

