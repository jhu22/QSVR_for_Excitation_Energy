#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
from pprint import pprint


# In[2]:


mat_file_path = 'qm7b.mat'
mat_data = scipy.io.loadmat(mat_file_path)


# In[5]:


print(len(mat_data['X']))


# In[7]:


print(len(mat_data['T']))


# In[4]:


pprint(mat_data)


# In[8]:


#test_CM = mat_data['X'][0]


# In[10]:


import numpy as np


# In[11]:


#eigenvalues, _ = np.linalg.eig(test_CM)


# In[12]:


#print(eigenvalues)


# In[13]:


#np.linalg.eig(mat_data['X'])


# In[14]:


from sklearn.decomposition import PCA


# In[26]:


pca_ = PCA(n_components = 6)


# In[24]:


input_x, _ = np.linalg.eig(mat_data['X'])


# In[25]:


#input_x.shape


# In[27]:


pca_.fit(input_x)


# In[28]:


qsvr_input_x = pca_.transform(input_x)


# In[29]:


#qsvr_input_x.shape


# In[30]:


import qiskit


# In[31]:


from qiskit_machine_learning.algorithms import regressors


# In[32]:


from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap


# In[33]:


from qiskit_machine_learning.kernels import QuantumKernel


# In[34]:


adhoc_dimension = 6


# In[35]:


adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")


# In[37]:


seed = 1376
algorithm_globals.random_seed = seed


# In[38]:


adhoc_backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed
)


# In[39]:


adhoc_kernel = QuantumKernel(feature_map=adhoc_feature_map, quantum_instance=adhoc_backend)


# In[40]:


qsvr_= regressors.QSVR(quantum_kernel=adhoc_kernel)


# The following quantum-derived properties describe the molecules: 1) PBE0 atomization energies, 2) zindo excitation energy, 3) zindo highest absorption intensity, 4) zindo homo, 5) zindo lumo, 6) zindo 1st excitation energy, 7) zindo ionization potential, 8) zindo electron affinity, 9) PBE0 homo, 10) PBE0 lumo, 11) GW homo, 12) GW lumo, 13) PBE0 polarizability A3, 14) SCS polarizability A3.

# In[43]:


Y_train =[]
for item in range(len(mat_data['T'])):
    property_ = mat_data['T']
    property_list = property_[item]
    excitation_energy = property_list[1]
    print(excitation_energy)
    excitation_enerty = [excitation_energy]
    Y_train.append(excitation_enerty)
Y_train = np.asarray(Y_train)


# In[44]:


Y_train.shape


# In[ ]:


qsvr_.fit(qsvr_input_x,Y_train)


# In[ ]:


y_predict = qsvr_.predict(qsvr_input_x)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(qsvr_input_x, Y_train, test_size=0.2, random_state=42)


# In[ ]:


qsvr_.fit(x_train,y_train)


# In[ ]:


y_predict_test = qsvr_.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


mse = mean_squared_error(y_test, y_predict_test)


# In[ ]:


r2 = r2_score(y_test, y_predict_test)


# In[ ]:


print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

