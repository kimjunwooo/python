#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def prediction(theta_0, theta_1,x):
    return theta_0 + theta_1 * x


# In[3]:


def prediction_difference(theta_0,theta_1,x,y):
    return prediction(theta_0,theta_1,x) - y


# In[4]:


def gradient_descent(theta_0,theta_1,x,y,num_iterations, alpha):
    m = len(x)
    cost_list = []
    
    for i in range(num_iterations):
        error = prediction_difference(theta_0,theta_1,x,y)
        cost = (error @ error) / (2*m)
        cost_list.append(cost)
        
        theta_0 = theta_0 - alpha * error.mean()
        theta_1 = theta_1 - alpha * (error * x).mean()
        
        if i % 10 == 0:
            plt.scatter(house_size, house_price)
            plt.plot(house_size,prediction(theta_0,theta_1,x), color = 'red' )
            plt.show
    
    return theta_0, theta_1, cost_list


# In[5]:


house_size = np.array([0.9,1.4,2,2.1,2.6,3.3,3.35,3.9,4.4,4.7,5.2,5.75,6.7,6.9])
house_price = np.array([0.3,0.75,0.45,1.1,1.45,0.9,1.8,0.9,1.5,2.2,1.75,2.3,2.49,2.6])

# 초기값 설정
th_0 = 2.5
th_1 = 0


# In[6]:


th_0, th_1, cost_list = gradient_descent(th_0, th_1, house_size, house_price, 200, 0.1)


# In[7]:


plt.plot(cost_list)


# In[ ]:




