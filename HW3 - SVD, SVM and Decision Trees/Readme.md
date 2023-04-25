# Homework 3: Singular Value Decomposition (SVD), Support Vector Machine (SVM) and Decision Trees  - Ishan Dane

## Sec. I. Abstract:

In this project, I conducted an analysis of the MNIST dataset using various machine learning techniques. The first part of the project involved performing an SVD analysis of the digit images, which required reshaping each image into a column vector. I analyzed the singular value spectrum to determine the necessary number of modes for good image reconstruction and interpreted the U, Σ, and V matrices. I also projected the data onto three selected V-modes using a 3D plot colored by digit label.

In the second part of the project, I built a classifier to identify individual digits in the training set using linear discriminant analysis (LDA). I successfully identified and classified two and three digits and quantified the accuracy of the separation for the most difficult and easiest pairs using LDA, support vector machines (SVM), and decision tree classifiers.

This project allowed me to gain hands-on experience with a popular dataset and various machine learning techniques, including SVD analysis and classification algorithms. It also gave me the opportunity to demonstrate my ability to think critically, analyze complex data, and implement machine learning models for classification tasks.


## Sec. II. Introduction and Overview:

In this project, I conducted an analysis of the popular MNIST dataset using various machine learning techniques. The MNIST dataset consists of 70,000 grayscale images of handwritten digits, and it serves as a benchmark dataset for classification tasks. As a software developer, working with datasets such as MNIST is crucial as it enables me to develop and improve my skills in data analysis, modeling, and implementation of machine learning algorithms.

In the first part of this project, I performed an SVD analysis of the digit images by reshaping them into column vectors. This analysis allowed me to interpret the U, Σ, and V matrices and determine the necessary number of modes for good image reconstruction. Additionally, I projected the data onto three selected V-modes using a 3D plot colored by digit label, which enabled me to visualize the data in a new way.

In the second part of the project, I built a classifier to identify individual digits in the training set using linear discriminant analysis (LDA). I successfully identified and classified two and three digits, quantified the accuracy of the separation for the most difficult and easiest pairs using LDA, support vector machines (SVM), and decision tree classifiers. These classification algorithms are widely used in the industry, and gaining experience with them is crucial for software developers.

Overall, this project allowed me to gain hands-on experience with a popular dataset and various machine learning techniques, including SVD analysis and classification algorithms. It also gave me the opportunity to demonstrate my ability to think critically, analyze complex data, and implement machine learning models for classification tasks, which are valuable skills in software development roles.


## Sec. III. Theoretical Background

The MNIST dataset and the techniques used in this project are rooted in various theoretical concepts that are fundamental to machine learning and data analysis. As a software engineer, understanding these concepts is crucial for developing and implementing effective machine learning models.

Singular value decomposition (SVD) is a matrix factorization technique used in various machine learning applications, including image processing and classification tasks. It is a powerful tool that enables us to analyze complex data and identify the most important features of a dataset. In this project, I used SVD to analyze the digit images in the MNIST dataset by reshaping them into column vectors and calculating the singular value spectrum. The resulting analysis allowed me to determine the necessary number of modes for good image reconstruction, which is essential for classification tasks.

Linear discriminant analysis (LDA) is a widely used classification algorithm that is commonly used in the industry. LDA involves projecting data onto a lower-dimensional subspace in such a way that maximizes the class separation. This algorithm has proven to be effective in various applications, including face recognition and handwritten digit recognition.

Support vector machines (SVMs) are another popular classification algorithm that has proven to be effective in various applications, including text classification, image classification, and bioinformatics. SVMs involve finding a hyperplane that separates the data into different classes, with a maximum margin between the hyperplane and the nearest data points of each class.
Decision trees are a classification algorithm that is widely used for decision-making tasks. Decision trees involve recursively splitting the data into smaller subsets based on a set of rules, with the aim of maximizing the information gain at each step. This algorithm is commonly used in various applications, including medical diagnosis and credit risk assessment.

Overall, the theoretical background behind the MNIST dataset and the techniques used in this project is essential knowledge for any software engineer interested in developing and implementing effective machine learning models.


## Sec IV. Algorithm Implementation and Development 

To begin, I started off by loading the MNIST dataset from the library **sklearn** using the function ```fetch_openml('mnist_784')```. This dataset consists of a collection of 70,000 images of handwritten digits from 0-9, each of size 28x28 pixels. Considering the size of the dataset and the technological resources available to me, I extracted the first 10,000 images and their corresponding labels from the dataset. To do this, I extracted the data attribute of the MNIST dataset through```mnist.data[:10000]``` which contained the pixel values of the images and ```mnist.target[:10000]``` which contained the labels for the images. Next I divided the images array by 255.0 to normalize the pixel values to the range [0, 1].

### SVD Analysis

### Part (a): 
The first step in my analysis of the MNIST dataset was to perform an SVD analysis of the digit images. To achieve this, I began by reshaping each image into a column vector using ```X = X.T```. This allowed me to create a data matrix where each column represents a different image.

Next, I applied the SVD function (np.linalg.svd()) from the NumPy linalg module to the data matrix. I set the full_matrices=False parameter to minimize the computational cost and memory requirements of the SVD computation. This parameter ensured that only the first min(m, n) columns of the left singular matrix U and the first min(m, n) rows of the right singular matrix V were computed, where m and n are the dimensions of the data matrix.
```
# Perform SVD on data matrix X_reshaped
U, S, V = LA.svd(X_reshaped, full_matrices=False)
```

Finally, I printed the shapes of the resulting matrices U, S, and V to verify their dimensions. This analysis allowed me to better understand the underlying structure of the dataset and paved the way for further exploratory analysis and modeling.

### Part (b):
For this task, I had to visualize the singular value spectrum and conclude how many modes were necessary for good image reconstruction. 

To investigate the singular value spectrum of the MNIST digit images, I used the SVD decompositions matrix ```S``` I had obtained previously.

The singular values are the values of the matrix S, and they represent the importance of each dimension (or feature) in the data matrix. By plotting the singular values, I can see how much information each dimension holds, and determine how many dimensions (or modes) are necessary for good image reconstruction.

My code creates a plot of the singular value spectrum using matplotlib, where the x-axis represents the index of each singular value, and the y-axis represents the value of each singular value. This plot can help me determine the rank r of the digit space, which represents the number of singular values necessary for good image reconstruction.

```
plt.plot(S)
```

### Part (c):
For this task, I was asked to interpret the U, Σ, and V matrices from the SVD decomposition.To do this, I created visualizations of the different matrices.

I plotted the V matrix using a 3D scatter plot. In the code, I used Matplotlib to create a new figure and added a 3D subplot. I then scatter plotted the first three rows of the V matrix. Each row of V represents a basis vector that defines a principal component of the digit images. 
```
ax.scatter(V[0,:], V[1,:], V[2,:], c='y', marker='o')
```
Next, I plotted the U matrix using another 3D scatter plot. In the code, I again created a new figure and added a 3D subplot. This time, I scatter plotted the first three columns of the U matrix. Each column of U represents a basis vector that defines a principal component of the digit images in the original space. 

```
ax.scatter(U[:, 0], U[:, 1], U[:, 2], c='r', marker='o')
```

### Part (d):
For task 4, I had to 3D plot the projection of the MNIST data X onto three V modes from the matrix ```V``` of right singular vectors, obtained through the SVD calculations.I first selected the three V-modes (columns) that I wanted to project onto, which were columns 2, 3, and 5 of the V matrix. Then I extracted these columns using slicing syntax and stored them in a variable called ```selected_v```.

Next, I projected the data matrix X_reshaped onto these selected V-modes using matrix multiplication. Specifically, I transposed X_reshaped and took the dot product with selected_v.

```
X_proj = np.dot(X_reshaped.T, selected_v)
```

 This gave me a matrix X_proj of shape (10000, 3), where each row corresponds to a different image and the three columns correspond to the projections onto the selected V-modes.
I then created a 3D plot using Matplotlib's Axes3D module to scatter plot the three columns of X_proj, where the color of each point was determined by its corresponding digit label Y. 

```
plot = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=Y.astype(int), cmap='rainbow')
```

### Building Classifiers

To begin, I had to split the dataset into training and testing sets. To do this, I used the ```.train_test_split()``` function from the ```sklearn.model_selection``` module. This function takes in the features of the dataset (mnist.data) and their corresponding labels (mnist.target) as arguments. I then specified the ```test_size``` parameter as 0.2, which means the function splits the data so that 20% of the data is used as the testing set. This function then returned 4 datasets:  X_train, X_test, Y_train and Y_test. Finally, I set the random_state parameter to 42. This was to ensure that the results were reproducible if I were to rerun the code in the future.


### Part (a):

For this task, I had to pick two digits and build a linear classifier (LDA) that can reasonably identify/classify them. I chose to build my linear classifier to identify digits - 9 and 5.

First, we created the training data by selecting all the instances with the digit 9 and 5 from the training set. To do this I used the ```np.where()``` function to find the indices of the relevant digits, and then extracted the corresponding data from the training set.
```
digit1_train_index = np.where(Y_train == '9')[0].tolist()
digit2_train_index = np.where(Y_train == '5')[0].tolist()

digit1_train_data = X_train.values[digit1_train_index , :]
digit2_train_data = X_train.values[digit2_train_index , :]
```
 I then concatenated these two sets of data to create a combined training set, along with the labels for each instance.
```
train_data = np.concatenate((digit1_train_data, digit2_train_data), axis=0)
train_index = digit1_train_index + digit2_train_index
train_labels = Y_train.values[train_index]
```

Next, I created the testing data by selecting all the instances with the digit 9 and 5 from the testing set. Once again, I used the ```np.where()``` function to find the indices of the relevant digits, and then extracted the corresponding data from the testing set, along with their respective labels.
```
train_index = digit1_train_index + digit2_train_index
train_labels = Y_train.values[train_index]
```
I then initialized a LinearDiscriminantAnalysis object and trained it on the training data using the ```.fit()``` function. 
```
lda = LinearDiscriminantAnalysis()
lda.fit(train_data, train_labels)
```
I then used the trained model to predict the labels for the training and testing data using the ```.predict()``` function.
```
y_pred = lda.predict(test_data)
y_pred2 = lda.predict(train_data)
```
Finally, I evaluated the performance of the model by calculating its accuracy on both the training and testing data using the ```.accuracy_score()``` function. The training accuracy was calculated by comparing the predicted labels for the training data to the true labels, while the testing accuracy was calculated by comparing the predicted labels for the testing data to the true labels.
```
print("Train Accuracy: ", accuracy_score(train_labels, y_pred2))
print("Test Accuracy: ", accuracy_score(test_labels, y_pred))
```

### Part (b):

For this task, I did the same thing as **Part (a)** except this time I had to pick three digits to build and test a linear classifier (LDA) on. I chose my digits to be - 4, 6 and 8. 

Just like I did in **Part (a)**, I extracted the necessary data for the digits from the datasets and created the training and test sets for the LDA. 
```
train_index = digit1_train_index + digit2_train_index + digit3_train_index
train_labels = Y_train.values[train_index]
test_data = X_test.values[test_index, :]
test_labels = Y_test.values[test_index]
```
Then I initialized another LinearDiscriminantAnalysis object and trained it on the training data. The model was then used to predict labels for the training and test data like previous and its accuracy for both sets were printed. 

### Part (c):
For this task, I had to conduct an analysis to determine which two digits in the MNIST dataset were the most difficult to separate. To do this, I used Linear Discriminant Analysis (LDA) to classify each combinational pair of digits and compared the accuracy of their classifiers. I started by creating a dictionary called ```dict``` to store the accuracy of each classifier.

I then used a nested loop to iterate through each pair of digits in the dataset. For each pair, I implemented a linear classifier that used LDA to project the training data onto a lower-dimensional space.
```
for i in range(0, 10):
   for j in range(i + 1, 10):
```
For each combinational pair, much like I did in **Part (b)** and **Part (a)**, I extracted and created the training and testing data corresponding to the digits. Once I had the training and testing data for the digit pair, I used LDA to fit a model to the training data and predict the labels for the testing data. I then calculated the accuracy score of each classifier and stored it in the dictionary previously created before printing out the results. 
```
dict[str(i) + str(j)] = accuracy_score(test_labels, y_pred)
           print("Digits: " + str(i) + " and " + str(j) + " Accuracy: ", accuracy_score(test_labels, y_pred))
```
From here, all I had to do to get the most difficult digits to separate was get the worst accuracy score in my dictionary. 
```
worst_digits = min(dict, key=dict.get)
print("Worst Digits: " + "(" + worst_digits[0] + ", " + worst_digits[1] + ")" + " Accuracy: ", dict[worst_digits])
```

### Part (d):
For this task, I had to find the easiest digits to separate. Since I already had a dictionary containing all the classifier accuracies for each digit pair, I just had to extract the best accuracy result.
```
best_digits = max(dict, key=dict.get)
print("Best Digits: " + "(" + best_digits[0] + ", " + best_digits[1] + ")" + " Accuracy: ", dict[best_digits])
```
### Part (e):
For this task, I had to explore how well SVM (support vector machines) and decision tree classifiers separate all ten digits from the MNIST dataset.

I created The SVM classifier using the ```.SVC()``` function from the ```sklearn.svm``` module  and trained it using the ```.fit()``` function on the X and Y training sets used previously. The trained SVM classifier was then used to predict the labels for the test set X_test, and the accuracy of the prediction was measured using the ```.accuracy_score()``` function, which compared the predicted labels with the actual labels in Y_test.

Similarly, I created a decision tree classifier using the ```.DecisionTreeClassifier()``` function from the ```sklearn.tree``` module and trained it using ```.fit()``` on the same training set. The trained decision tree classifier was then used to predict the labels for the test set, and the accuracy of the prediction was again measured using the ```.accuracy_score()``` function. I then plotted the tree structure using the ```.plot_tree()``` function from ```sklearn.tree```.


### Part (f):
For this task, I had to Compare the performance between LDA, SVM and decision trees on the hardest and easiest pair of digits to separate (from above). 

To compare the performance of LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate, I selected digits 5 and 8 as the hardest pair from my previous calculations. Firstly, I split the data into training and testing sets for these digits for those specific digits using the ```np.where()``` function to extract data specific to the digits like I did previously. 

Then, I trained an SVM classifier and a decision tree classifier on the training data and made predictions on the testing data using each classifier. The accuracy of the predictions was calculated using the accuracy_score function and printed out for each classifier. For the LDA classifier, I used the accuracy value stored in the dictionary for the worst pair of digits obtained in a previous step.

For the performance on the easiest digits, I selected digits 6 and 7 as the easiest pair from my previous calculations. I then repeated the processor done for the hardest digits and printed out the accuracy scores for each classifier. 

## Sec. V. Computational Results

## SVD Analysis

### Part (a)
For this task, I computed the SVD for the training and label set of the first 10000 images from the MNIST data set. The SVD computation returned U, S and V matrices.
### Part (b)
For this task, I plotted the singular values from the SVD decomposition to find the number of modes necessary for good image reconstruction. The graph is shown below:

![Screenshot 2023-04-24 at 11 22 36 PM](https://user-images.githubusercontent.com/122940974/234191275-75021465-0a66-4cc8-94c3-f695f8b5948b.png)

From the graph, We can see that we only need around 35 modes to reconstruct a good image. We can also see that the elbow point occurs at around Singular Value Index = 35 ~ (Mode 35) and therefore the rank r of digit space is 35.

### Part (c)
For this task, I was asked to interpret the U, V and Σ matrices. The U and V matrices from the SVD represent the left and right singular vectors, respectively, while the Σ matrix contains the singular values. The left singular vectors in the U matrix represent the directions in the original data space that capture the most variation in the data. The right singular vectors in the V matrix represent the directions in the image space that capture the most variation in the images. The singular values in the Σ matrix represent the importance of each of these directions and thus contain the scaling factors for these basis vectors. The graphs of U and V are shown below:

![Screenshot 2023-04-24 at 11 23 13 PM](https://user-images.githubusercontent.com/122940974/234191377-9b82547c-3518-4703-ac2e-153de2e533ad.png)

From this graph, we can see that the points are all clustered around a common location for all 3 V components. The fact that the points are clustered around a center indicates that there is some commonality among the features in the dataset, as represented by the singular vectors showing a high degree of correlation between these features in the dataset.

![Screenshot 2023-04-24 at 11 23 41 PM](https://user-images.githubusercontent.com/122940974/234191449-b7197bf9-3b92-423c-9e76-ad2af7f03eb1.png)

From this graph, we can see that the points are relatively spread out evenly. This shows that the U scatter plot has more variation or spread in the data compared to the V scatter plot. This could mean that the left singular vectors (represented by U) are capturing more of the variability or structure in the data, while the right singular vectors (represented by V) are clustering around a center because they are representing the least variable or informative directions. This could indicate that the first few left singular vectors are the most important for explaining the variability in the data.

### Part (d)
For this task, I plotted the projection of the MNIST data onto the V-columns 2, 3 and 5. The graph is shown below:

![Screenshot 2023-04-24 at 11 24 05 PM](https://user-images.githubusercontent.com/122940974/234191527-33b67fe9-4167-4965-9fa5-a70f5216a7dd.png)

Each point in the plot represents a digit from the dataset, and its color indicates the true class label of the digit. The points seem to be centered around a common location between the V-columns. This suggests that the projected data has lower dimensionality than the original data, which could be an indication that the chosen subspace captures the most important features of the data. 


## Building Classifiers

### Part (a)
For the first task in building classifiers, I picked out two digits - 9 and 5 - and built a linear classifier to correctly identify them in a test set of 9s and 5s. The accuracy of the classifier is shown below:
```
Train Set Accuracy:  0.9882775571941766
Test Set Accuracy:  0.9851466765688823
```

### Part (b)
For this task, I repeated the process of **Part (a)** except this time building a linear classifier to correctly identify digits - 4, 6 and 8 - from a test set of the digits. The accuracy of the classifier in identifying the digits is shown below:
```
Train Set Accuracy:  0.9777265278873581
Test Set Accuracy:  0.9757905138339921
```

### Part (c)
For this task, I had to find the two digits that were the most difficult to separate. To do this, I built a linear classifier for every combinational pair of digits and compared their accuracies. The digits with the worst accuracy are show below:
```
Worst Digits: (5, 8) Accuracy:  0.9498098859315589
```
### Part (d)
For this task, I used my previous dictionary of linear classifier accuracy values for each pair of digits to find the most easily separated digits. The digits are shown below:
```
Best Digits: (6, 7) Accuracy:  0.9968954812004139
```
### Part (e)
For this task, I built and compared the performance between a SVM classifier to a Decision Tree classifier in classifying all 10 digits from the dataset. The accuracy results are shown below:
```
SVM Accuracy:  0.9764285714285714
Decision Tree Accuracy: 0.8688571428571429
```
The plot for the decision tree structure is shown below:

![Screenshot 2023-04-24 at 11 24 36 PM](https://user-images.githubusercontent.com/122940974/234191608-22899bd2-385f-4c78-a5d3-0ee7e63a7036.png)

We can see how expansive the decision tree structure can get as it forms relationships between the position of all the different points. The accuracy scores also show us that the SVM classifier does a way better job at correctly identifying each of the digits compared to the Decision Tree. 

### Part (f)
For this task, I had to compare the performance between the Linear Classifier (LDA), SVM and Decision Tree on classifying the hardest and easiest digits. The results of each of the classifiers are shown below.

On the hardest digits - 5 and 8
```
SVM Accuracy:  0.988212927756654
Decision Trees Accuracy:  0.9520912547528517
LDA Accuracy:  0.9498098859315589
```
The plot of the decision tree structure for digits - 5 and 8 - is shown below:

![Screenshot 2023-04-24 at 11 25 04 PM](https://user-images.githubusercontent.com/122940974/234191699-27d91350-a1af-45a2-a532-a7c4209fd7bb.png)

On the easiest digits - 6 and 7
```
SVM Accuracy:  0.9993101069334254
Decision Trees Accuracy:  0.9934460158675406
LDA Accuracy:  0.9968954812004139
```
The plot of the decision tree structure for digits - 6 and 7 - is shown below:

![Screenshot 2023-04-24 at 11 25 22 PM](https://user-images.githubusercontent.com/122940974/234191759-4a7616b4-530f-4c0b-8b70-295b9f8ccb23.png)


## Sec. VI. Summary and Conclusion

In this project, I explored the MNIST dataset and built different classifiers - SVM, LDA and Decision Trees - to analyze and predict data. I started by scaling the dataset and applying SVD to reduce the dimensionality of the data and obtain the decomposition matrices U, V and S.  I applied dimensionality reduction to the data and visualized it in three dimensions. By selecting a subset of the principal components and projecting the data onto them, I was able to observe distinct clusters of data points in the projected space, showing that most of the variance in the data could be explained by a few principal components.

I then explored the performance of various classification algorithms, such as SVM, decision trees, and LDA, on the dataset. By comparing the accuracy of these algorithms on classifying different digits, I found that SVM performed the best in every situation when compared to LDA and Decision Trees. 

Overall, in this project, I demonstrated proficiency in various machine learning techniques, including data preprocessing, dimensionality reduction, and classification. The ability to effectively analyze and visualize complex datasets is a valuable skill in the field of software engineering, and the techniques presented in this project can be applied to a wide range of applications, such as image recognition and natural language processing.
