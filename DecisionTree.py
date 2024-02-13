import pandas as ps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
classifier = DecisionTreeClassifier()
classifier.fit(iris.data, iris.target)
plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled=True)
plt.show()