
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np


# In[42]:


data_train = pd.read_csv('train_set_use_keep_time', sep='\t', header=None, names=['cid','did', 'time', 'label'])


doc_info = pd.read_csv('train_nid_data_content_length', sep='\t', header=None, names = ['did','cate','url','site','author','author_level','title','content_len','key_word','tag1','tag2','pic_num','pic_att1','pic_att2','pic_att3','pic_att4','pic_att5'])

#### 现在需要讲两者json 进来
merge = pd.merge(data_train, doc_info, on='did', how='left')
merge.to_csv('merge.csv', index =False, sep='\t')

