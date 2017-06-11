
# coding: utf-8

# In[ ]:

###Example....
#>>>python test.py dp.arg3*=4 dp.arg2='terry'
#


# In[9]:

from unsupported_dan.cl_args import easy_args
import sys
from easydict import EasyDict as edict


# In[10]:

easy_args


# In[8]:

sysArgs = sys.argv
dp = edict({})

dp.arg2='gary'
dp.arg3=2.0

print(dp.arg2, dp.arg3)
easy_args(sysArgs, dp)
print(dp.arg2, dp.arg3)


# In[ ]:



