Authors: David Pogorzelski and Lucas Porz, Helmholtz-Zentrum Hereon, 2026

Scripts and data to generate the surface OC contents of the Baltic Sea using a deep neural network. A hyperparameter search using the Hydra framework is used to find the optimal model parameters.

/corg/corg_train_predict.ipynb is the main interactive Jupyter Notebook which runs the hyperparameter search to find the best model parameters based on comparison to the testing data points, then retrains the model with the total dataset using specified parameters, and finally predicts OC contents on a 500 m x 500 m grid.

/data contains the input data 
/corg contains the necessary python scripts
/corg/outputs will contain the final prediction as a CSV file
/corg/multirun contains the results of the hyperparameter search
/corg/models and /corg/misc contain the currently trained PyTorch model
/corg/plots contains some plots generated during training and prediction

