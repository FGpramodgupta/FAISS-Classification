#!/usr/bin/env python
# coding: utf-8

# In[19]:


## Load Dataaset

import glob
import os
import pandas as pd


# In[20]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


# In[21]:


document_text = []
document_metadata = []
document_id = []
document_embeddings = []


# In[22]:


for folder_path in glob.glob("dataset/train/*"):
    print(folder_path)
    
    category = "TAX"
    if folder_path in ["dataset/train\pancard","dataset/train\passport"]:
        print('India')
        category = "KYC"
        
    for path in glob.glob(folder_path + "/*.txt"):
        print(path)
        filename = os.path.split(path)[1]
        filename_without_extension = filename[:-4]
        file_metadata = filename_without_extension.split('_')
        metadatas = {
            "title" : file_metadata[0],
            "sequence" : file_metadata[1],
            "page_number" : file_metadata[2],
            "category" : category
        }

        
        with open(path,'r',encoding="UTF-8") as f:
            ocr_text = f.read()
#             print(ocr_text)
            document_text.append(ocr_text)
        
            embedding = model.encode(ocr_text).tolist()
            #document_embeddings.append(embedding)
        document_metadata.append(metadatas)
        document_id.append(filename_without_extension)


# In[ ]:





# In[23]:


input_sample = {
    'documents_text' : document_text,
    'metadatas' : document_metadata,
    #'embeddings' : document_embeddings,
    'ids' : document_id
}
input_sample


# In[ ]:





# In[24]:


df = pd.DataFrame.from_dict(input_sample)
df


# In[27]:


from sentence_transformers import SentenceTransformer
text = df['documents_text']
print(text)
encoder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = encoder.encode(text)
vectors


# In[28]:


import faiss

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)


# In[ ]:





# In[29]:


import numpy as np

search_text = r"""star faa Ana Gare
INCOME TAX DEPARTMENT & GOVT. OF INDIA
REKHA SANDEEP SIRSWAL :

AJITKUMAR DULGACH
24/09/1986

Permanent Account Number

DOUPS6641A

gee

"""

search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)


# In[ ]:





# In[30]:


k = index.ntotal
distances, ann = index.search(_vector, k=k)
distances, ann


# In[31]:


labels  = df['metadatas']
category = labels[ann[0][0]]


# In[32]:


category


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


title_predicted_list=[]


# In[35]:


for folder_path in glob.glob("dataset/test/*"):
    print(folder_path)
#     if folder_path != "dataset/test\w9":
#         continue
#     else:
#         prediction_category = "KYC"
    for path in glob.glob(folder_path + "/*.txt"):
        print(path)
        
        
        with open(path,'r',encoding="UTF-8") as f:
            ocr_text = f.read()    

            search_vector = encoder.encode(ocr_text)
            _vector = np.array([search_vector])
            faiss.normalize_L2(_vector)
            k = index.ntotal
            distances, ann = index.search(_vector, k=k)
            labels  = df['metadatas']
            category = labels[ann[0][0]]
            title_predicted_list.append(category)
            print(category)


# In[36]:


len(title_predicted_list)


# In[37]:


title_predicted_list.count("passport")


# In[38]:


188 + 27


# In[ ]:





# In[ ]:




