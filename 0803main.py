import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Fitting the model using Bayesion Estimator
from pgmpy.estimators import BayesianEstimator
# Defining the Bayesion Model structure
from pgmpy.models import BayesianNetwork

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
flood_test2_1 = pd.read_csv('./flood_test.csv')
# flood_test1 = np.array(flood_test1)
flood_test2_1 = np.array(flood_test2_1)
# Shortcut for learning all the parameters and adding the CPDs to the model.

flood_model.fit(
    data=flood_train,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

values = pd.DataFrame(index=range(1),
                      columns=['Elev','Popd','Impe','Ra','GDP','Road','Rivd','Driv','Pipd','Dpip','Slop'])

values.iloc[:, :] = flood_test2_1[:, :]
y_pred = flood_model.predict_probability(values)
y_pred.to_csv('Fld_pred.csv', index=False)
