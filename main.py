import pandas as pd
data = [['Where are your headquarters located?', 'location'],
['Throw my cellphone in the water', 'random'],
['Network Access Control?', 'networking'],
['Address', 'location']]
df = pd.DataFrame(data, columns = ['text', 'category'])


import faiss


from sentence_transformers import SentenceTransformer
text = df['text']
encoder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = encoder.encode(text)
vectors


import faiss

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)



import numpy as np

search_text = 'where is your office?'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)


k = index.ntotal
distances, ann = index.search(_vector, k=k)
print(distances, ann)

results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

merge = pd.merge(results,df,left_on='ann',right_index=True)

labels  = df['category']
category = labels[ann[0][0]]

print(category)


