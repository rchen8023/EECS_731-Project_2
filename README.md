# EECS_731-Project_2
In this project, I used three classification model to train and predict Shakespeare's plays and determine the players. I used Gaussian Naive Bayes, Random Forest Classification, Decission Tree Classificaiton. Then I tried to modify the model and some data features for higher performance with Random Forest Classification. 

# Project Instruction
Classy Shakespeare plays and players

1. Set up a data science project structure in a new git repository in your GitHub account
2. Download the Shakespeare plays dataset from https://www.kaggle.com/kingburrito666/shakespeare-plays
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5.  one or more classification models to determine the player using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

# Datasets
I got the datasets from https://www.kaggle.com/kingburrito666/shakespeare-plays 

# Results
The result shows that 22% accuracy for Gaussian Naive Bayes, 81% for Random Forest, 79% for Decission Tree. And the data without 'Line' feature would get higher performance in Random forest. 

# References: 
https://stackoverflow.com/questions/17114904/python-pandas-replacing-strings-in-dataframe-with-numbers 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html 
https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html 
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html 
https://scikit-learn.org/stable/modules/naive_bayes.html 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
