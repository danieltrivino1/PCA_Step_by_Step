# Libraries
import numpy as np
from numpy.core.fromnumeric import mean, shape, std
from numpy.lib.function_base import cov
from numpy.linalg import eig
from numpy.typing import _128Bit
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Import data from Excel
df = pd.read_excel(r'C:\Users\dpedr\OneDrive\Documents\Personal\Projects\principal_component_analysis\pca_data.xlsx', sheet_name = 'data')
data = np.array(df.T, dtype=np.float64)

# Variables
means=[]
devs=[]

# Calculate mean column and standard deviation column
for i in range(len(data)):
    mean_id = np.mean(data[i]) # Calculate mean
    dev_id = np.std(data[i],ddof=1) # Calculate sample std deviation
    means.append(mean_id)
    devs.append(dev_id)

# Standardize the data
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = ((data[i][j] - means[i]) / devs[i])

# Covariance matrix computation
cov_matrix = np.cov(data,rowvar=True,ddof=1)

# Plot covariance with Plotly Heatmap
x=['Price', '#Rooms', '#Bedrooms', '#Bathrooms', '#CarSpots', 'BuildingArea', 'DistanceDowntown', 'YearBuilt']
fig = ff.create_annotated_heatmap(np.round(cov_matrix,6), x=x, y=x, colorscale='blues', showscale=True)
fig['layout']['yaxis']['autorange'] = "reversed" # Get features coordinates to match on the matrix 
fig.show()

eig_val, eig_vec = np.linalg.eigh(cov_matrix) # Returns eigenvectors of norm 1

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))] # Create a list of tuples with (Eigenvalues, Eigenvectors)

eig_pairs.sort(key=lambda x: x[0], reverse=True) # Sort list of tuples in decreasing order by the eigenvalues

# Scree Plot
eig_val = [np.abs(eig_val[i]) for i in range(len(eig_val))]
eig_val = np.sort(eig_val)[::-1]
fig = px.line(x=np.linspace(1,len(eig_val),len(eig_val)),y=eig_val, labels={'x':'Component Number','y':'Eigenvalue'}, title='Scree Plot')
fig.show()

# Variables
k_matrix=[]
k=3

# Data Transformation
for i in range(k):
    k_matrix.append(eig_pairs[i][1])

k_matrix = np.array(k_matrix)

results_matrix = np.matmul(k_matrix,data).T

# 3D scatter plot
fig = px.scatter_3d(pd.DataFrame(results_matrix), x=0,y=1,z=2)
fig.show()