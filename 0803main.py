import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Fitting the model using Bayesion Estimator
from pgmpy.estimators import BayesianEstimator
# Defining the Bayesion Model structure
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.readwrite import BIFWriter
from pgmpy.estimators import MaximumLikelihoodEstimator

# Defining the model structure. We can define the network by just passing a list of edges.

flood_model = BayesianNetwork([('Ra', 'Fld'),
                               ('GDP', 'Fld'),
                               ('Popd', 'GDP'), ('Popd', 'Fld'), ('Popd', 'Pipd'), 
                               ('Pipd', 'Fld'), ('Pipd', 'Dpip'),
                               ('Dpip', 'Fld'),
                               ('Road', 'Pipd'), ('Road', 'Fld'),
                               ('Impe', 'Pipd'), ('Impe', 'Fld'),
                               ('Elev', 'Fld'), ('Elev', 'Slop'),
                               ('Slop', 'Fld'),
                               ('Rivd', 'Fld'), ('Rivd', 'Driv'), ('Rivd', 'Slop'),
                               ('Driv', 'Fld')])

# Plot the network.
nx.draw_circular(
    flood_model, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
)
plt.show()
# training data
flood_train = pd.read_csv('./flood_train.csv')
# test data
# flood_test1 = pd.read_csv('./flood_test1.csv')
flood_test2_1 = pd.read_csv('./city_test_try.csv')
# flood_test1 = np.array(flood_test1)
flood_test2_1 = np.array(flood_test2_1)
# Shortcut for learning all the parameters and adding the CPDs to the model.

flood_model.fit(
    data=flood_train,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)
flood_model.save('flood_model.bif', filetype='bif')

# predict procedure
#ii = 0
#row = 0
#values = pd.DataFrame(index=range(788),
#                      columns=['Elev','Popd','Impe','Ra','GDP','Road','Rivd','Driv','Pipd','Dpip','Slop'])

#values.iloc[:, :] = flood_test1[:, :]
#y_pred = flood_model.predict_probability(values)
#y_pred.to_csv('Fld_pred_city1.csv', index=False)

values = pd.DataFrame(index=range(1),
                      columns=['Elev','Popd','Impe','Ra','GDP','Road','Rivd','Driv','Pipd','Dpip','Slop'])
#for i in range(1):
#    print('ii')
#    row = i*100
#    values.iloc[0:100, :] = flood_test[row:row + 100, :]
#    y_pred = flood_model.predict_probability(values)
#    y_pred.to_csv('Fld_pred_city5.csv', index=False, mode='a')
#    ii = ii + 1
values.iloc[:, :] = flood_test2_1[:, :]
y_pred = flood_model.predict_probability(values)
y_pred.to_csv('Fld_pred_citytest_try.csv', index=False)