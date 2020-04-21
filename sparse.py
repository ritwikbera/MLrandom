import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')

inputs = np.random.binomial(1, 0.1, 20000000).reshape(2000,10000)
outputs = np.random.binomial(1, 0.5, 2000)

plt.spy(inputs)
plt.title('Is it sparse ?')
plt.show()

sparse_inputs = csr_matrix(inputs)

# checking the effects of sparsity on compression
import seaborn as sns

dense_size = np.array(inputs).nbytes/1e6
sparse_size = (sparse_inputs.data.nbytes + sparse_inputs.indptr.nbytes + sparse_inputs.indices.nbytes)/1e6

sns.barplot(['DENSE', 'SPARSE'], [dense_size, sparse_size])
plt.ylabel('MB')
plt.title('Compression')
plt.show()