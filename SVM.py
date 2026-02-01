# SVM
# pros: high dimensional space, use different kernel include linearm, polynomial, radial basis funcition and sigmoid for non lineard decision boundaries, fewer assuption than naive bayes or linear regression 
# SVMs can also be extended to handle multi-class classification tasks through techniques like One-vs-One or One-vs-All, where multiple SVM models are trained to distinguish between each pair of classes or between each class and the rest, respectively.
# In addition to classification, SVMs can be used for regression tasks by fitting a hyperplane within a specified margin of error around the training data. This variant is known as Support Vector Regression (SVR).
# use case : multiclass classification - 1 vs 1 (red vs blue ) 1 vs rest (Red vs. [Blue & Green])

# # sample case- Linear SVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np



# #1. generate test data 
# from sklearn.datasets import make_blobs
# x,y = make_blobs(n_samples=10, n_features=2, centers=2, random_state=0)
# # 10 data sets, 2 feature x1 x2 for each data and in 2 different lables (y)
# print(x)
# print(y)

# [[ 1.12031365  5.75806083]
#  [-0.49772229  1.55128226]
#  [ 1.9263585   4.15243012]
#  [ 2.49913075  1.23133799]
#  [ 3.54934659  0.6925054 ]
#  [ 1.7373078   4.42546234]
#  [ 2.91970372  0.15549864]
#  [ 2.84382807  3.32650945]
#  [ 0.87305123  4.71438583]
#  [ 2.36833522  0.04356792]]
# [0 1 0 1 1 0 1 0 0 1]

# import matplotlib.pyplot as plt

# dx, dy = make_blobs(n_samples=500, n_features=2, centers=2,random_state=0)

# plt.scatter(dx.T[0], dx.T[1], c=dy, cmap='Dark2')
# plt.grid(True)
# plt.show()

# 2. standardize test data : dx -> sx_std

#Make normal distribution : standardscalar()- avg =0, var=1 


# from sklearn.preprocessing import StandardScaler

# dx, dy = make_blobs(n_samples=500, n_features=2, centers=2,random_state=0)
# dx_std = StandardScaler().fit_transform(dx)

# plt.scatter(dx_std.T[0], dx_std.T[1], c=dy, cmap='Dark2')
# plt.grid(True)
# plt.show()

# 3. cut train test 0.8 0.2  


# from sklearn.model_selection import train_test_split

# # dx, dy = make_blobs(n_samples=500, n_features=2, centers=2,random_state=0)
# # dx_std = StandardScaler().fit_transform(dx)
# # dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std,dy, test_size=0.2, random_state=0)

# # print(dx.shape)
# # print(dx_train.shape)
# # print(dx_test.shape)
# # print(dy.shape)
# # print(dy_train.shape)
# # print(dy_test.shape)

# # 4. SVC- c: classifier

# from sklearn.svm import LinearSVC

# dx, dy = make_blobs(n_samples=500, n_features=2, centers=2,random_state=0)
# dx_std = StandardScaler().fit_transform(dx)
# dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std, dy, test_size=0.2, random_state=0)

# linear_svm = LinearSVC(dual='auto', random_state=0)

# # random state =0 -> reproducibility 
# # solver logic : primal - solve for weight on a lot of samples , dual: solve on relationship if a lot features
# linear_svm.fit(dx_train, dy_train)

# #4.1 Manual Boundary calculation
# # think of x0 as x-coordinates, to get corresponding y-coordinates , plut xo into line equaiton. move x1 to one side as y
# # w0*x0 + w1 *x1 +b =0 => x1 = (-w0/w1) * x0 - (b/w1)
# # The Decision:
# # If $w_0x_0 + w_1x_1 + b > 0$, the model predicts Class A.
# # If $w_0x_0 + w_1x_1 + b < 0$, the model predicts Class B.
# # If it equals exactly 0, you are standing directly on the Decision Boundary.

# #weight, bias
# w = linear_svm.coef_[0]
# b= linear_svm.intercept_[0]

# # find max and min of data and create line 
# x0= np.linspace (dx_std[:,0].min(), dx_std[:, 0].max(), 200)
# decision_boundary = -w[0]/w[1] * x0 - (b/w[1])

# # calculate margins
# # the distance between the boundary 

# predictions = linear_svm.predict(dx_test)
# print(linear_svm.score(dx_train, dy_train))
# print(linear_svm.score(dx_test, dy_test))

# margin = 1 / w[1]
# gutter_up = decision_boundary + margin
# gutter_down = decision_boundary - margin

# # 4. Plotting
# plt.figure(figsize=(10, 6))
# plt.scatter(dx_train[:, 0], dx_train[:, 1], c=dy_train, cmap='Dark2', label='Train Data')

# plt.plot(x0, decision_boundary, 'k-', label='Decision Boundary')
# plt.plot(x0, gutter_up, 'k--', alpha=0.5, label='Margin')
# plt.plot(x0, gutter_down, 'k--', alpha=0.5)

# plt.ylim(dx_std[:, 1].min() - 0.5, dx_std[:, 1].max() + 0.5)
# plt.title(f"LinearSVC Boundary (Test Score: {linear_svm.score(dx_test, dy_test):.2f})")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# 0.9675
# 0.96

# #nonlinear SVM
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_circles
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler

# # 1. Create Non-linear Data (Circles)
# # noise : how far the data spread out 
# X, y = make_circles(n_samples=500, factor=0.3, noise=0.1, random_state=0)
# X = StandardScaler().fit_transform(X)

# # 2. Fit RBF SVM (The most popular non-linear kernel)
# # C: Penalty for misclassification
# # gamma: How far the influence of a single training point reaches to the decision boundaries
# # C =1 (High, small can be 0.01): penalty - Inverse of degree of regularization ( tolerate deviation )- smaller c to prevent overfit
# clf = SVC(kernel='rbf', C=1.0, gamma=0.5)
# clf.fit(X, y)

# # 3. Create a Meshgrid to plot the boundary
# h = .02  # step size in the mesh
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# # 4. Predict across the entire meshgrid
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # 5. Plotting
# plt.figure(figsize=(8, 6))
# # Draw the filled decision regions
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
# # Draw the actual boundary line
# plt.contour(xx, yy, Z, colors='k', linewidths=1)
# # Plot the data points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

# plt.title("Non-Linear SVM with RBF Kernel")
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

# # 1. Create 'Moon' data (harder to separate than blobs)
# X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
# X = StandardScaler().fit_transform(X)

# # 2. Define our Kernels to compare
# kernels = [
#     ("Polynomial (Degree 3)", SVC(kernel='poly', degree=3, C=5, coef0=1)),
#     ("Polynomial (Degree 5)", SVC(kernel='poly', degree=5, C=5, coef0=1)),
#     ("RBF (Gaussian)", SVC(kernel='rbf', gamma=1, C=5))
# ]

# # 3. Setup Plotting Mesh
# h = .02
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# plt.figure(figsize=(18, 5))

# for i, (title, clf) in enumerate(kernels):
#     clf.fit(X, y)
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
#     plt.subplot(1, 3, i + 1)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
#     plt.title(title)
#     plt.xticks([]); plt.yticks([])

# plt.tight_layout()
# plt.show()


# #deg3- smooth parabola or curbic curve
# #deg 5: wiggy curve and might overfit

## BestHyperparameterCis0.01_Linear_SVM- 1000 iteration for the best solution
# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.svm import LinearSVC
# import numpy as np
# import matplotlib.pyplot as plt

# dx, dy = load_breast_cancer(return_X_y=True)
# dx_std = StandardScaler().fit_transform(dx)
# dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std, dy, test_size=0.2, random_state=0)

# cv_scores = []
# test_scores = []
# x = [10 ** n for n in range(-4, 5)]
# x_str = [str(n) for n in x]

# for c in x:
#   linear_svc = LinearSVC(C=c, max_iter=10000).fit(dx_train, dy_train)
#   cv_scores.append(cross_val_score(linear_svc, dx_train,dy_train, cv=5).mean())
#   test_scores.append(linear_svc.score(dx_test, dy_test))

# plt.title('Linear SVM hyperparameter')
# plt.plot(x_str, cv_scores, label='CV score')
# plt.plot(x_str, test_scores, label='Test score')
# plt.xlabel('C')
# plt.ylabel('accuracy (%)')
# plt.legend()
# plt.grid(True)
# plt.show()



# non linear SVM with C,gammam, kernel parameter 

# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# import matplotlib.pyplot as plt

# dx, dy = load_breast_cancer(return_X_y=True)
# dx_std = StandardScaler().fit_transform(dx)
# dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std, dy, test_size=0.2, random_state=0)

# x = [10 ** n for n in range(-2, 3)]

# param_grid = {'C': x,
#          'gamma': x,
#          'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

# model = GridSearchCV(SVC(), param_grid)
# model.fit(dx_train, dy_train)

# print('Best params: ', model.best_params_)
# print('CV score:', round(model.best_score_, 3))
# print('Test score:', round(model.score(dx_test, dy_test),3))


# Best params:  {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
# CV score: 0.982
# Test score: 0.982


# non linear SVM but randomizedsearch() over max_iter 

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt

dx, dy = load_breast_cancer(return_X_y=True)
dx_std = StandardScaler().fit_transform(dx)
dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std, dy, test_size=0.2, random_state=0)

param_grid = {'C': np.linspace(1, 100, 100),
        'gamma': np.linspace(0.01, 1, 100),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
model = RandomizedSearchCV(SVC(), param_grid, n_iter=100)
model.fit(dx_train, dy_train)

print('Best params:', model.best_params_)
print('CV score:', round(model.best_score_, 3))
print('Test score:', round(model.score(dx_test, dy_test),3))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get predictions
predictions = model.predict(dx_test)

# Generate the matrix
cm = confusion_matrix(dy_test, predictions)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Breast Cancer Detection")
plt.show()
# Best params: {'kernel': 'rbf', 'gamma': np.float64(0.01), 'C': np.float64(23.0)}
# CV score: 0.978
# Test score: 0.982