# Homework 2: Singular Value Decomposition and PCA - Ishan Dane

## Sec. I. Abstract:
In this project, I worked with the Yale Face Database, which contains 2414 gray-scale images of 39 individuals under different lighting conditions. I used Python and various other data science libraries such as NumPy ,SciPy and Matplotlib to perform exploratory data analysis, visualizations and apply linear algebra techniques to the data. 

First, I computed a correlation matrix between the first 100 images in the database and visualized the matrix through a heatmap. Using my correlation matrix, I then identified the two most highly correlated and most uncorrelated images and plotted them to visualize the correlations. Next, I repeated this process with 10 provided images to create a 10x10 correlation matrix and plot it as a heatmap like previously. This was important as it provided me with information on the correlations between images on a smaller data scale. 

Next, I created a matrix Y by using the dot product on the data matrix X with its transpose (X.T) and found the top six eigenvectors with the largest magnitude eigenvalues. I then performed singular value decomposition (SVD) on X to find the first six principal component directions and calculated the percentage of variance captured by each of them. Finally, from here, I compared the first eigenvector from the eigenvalue decomposition with the first SVD mode and computed the norm of their difference in absolute values. To visualize the principal component directions , I also plotted the first six SVD modes.

Overall, this project allowed me to apply Correlation Analysis, Singular Value Decomposition (SVD) and Principal Component Analysis(PCA) techniques to a real-world dataset and gain experience working with large, high-dimensional data.

## Sec. II. Introduction and Overview:

In this project, I applied various machine learning techniques to analyze the Yale Face Database. The dataset consisted of 2414 downsampled gray-scale face images, each of size 32x32 pixels. The goal of the project was to explore different methods of data analysis and feature extraction to gain insights into the nature of the images.

To begin, I computed a correlation matrix between the first 100 images in the dataset. By computing the dot product (correlation) between each pair of images, I was able to construct a 100x100 matrix of correlations, which I visualized using the pcolor function in Python's matplotlib library. This allowed me to identify the pairs of images that were most highly correlated and least correlated, which I then plotted to compare and contrast the visual similarities and differences between these faces.

Next, I computed a 10x10 correlation matrix between 10 given images from the dataset and visualized it using pcolor again. This gave me a more granular view of the correlations between the images and allowed me to identify patterns and structures that were not as readily apparent in the larger correlation matrix.

To further analyze the dataset, I created a matrix Y = XXT, where X is the matrix of face images and X.T is its transpose. From this, I found the first six eigenvectors with the largest magnitude eigenvalue. I also applied the Singular Value Decomposition (SVD) to X to find the first six principal component directions. I then compared the first eigenvector from the eigenvalue decomposition with the first SVD mode, computing the norm of the difference between their absolute values. Finally, I computed the percentage of variance captured by each of the first six SVD modes and plotted them to gain a visual understanding of how much information each mode contained. 

This project showcases my skills in data analysis, matrix manipulation, and feature extraction, as well as my proficiency in Python programming and various machine learning libraries. By applying these techniques to a real-world dataset, I gained valuable insights into the nature of facial images and their correlations, which can be useful in various machine learning applications such as facial recognition and emotion detection.

## Sec. III. Theoretical Background

The theoretical background of this project lies in several key areas of machine learning. One such area is the computation of correlation matrices, which are frequently used in data analysis and feature extraction. In this project, we compute correlation matrices between sets of images to identify patterns and relationships between them. We also make use of eigenvectors and eigenvalues, which are fundamental concepts in linear algebra and can be used to decompose matrices into their component parts. By computing the eigenvectors and eigenvalues of the correlation matrix and the matrix Y = XXT, we are able to identify the most important features and patterns in the image data.

Another important concept in this project is the singular value decomposition (SVD), which is a powerful tool for dimensionality reduction and feature extraction. By computing the SVD of the matrix X, we can identify the most important features in the data and reduce the dimensionality of the data while preserving as much of the original information as possible. This allows us to more effectively analyze and work with large datasets.

In summary, the understanding of these concepts is essential for developing and implementing effective machine learning algorithms, and for interpreting the results obtained from them. The concepts of correlation matrices, eigenvectors, and singular value decomposition were applied to analyze the Yale Faces dataset in this project, highlighting their practical significance in the context of data analysis and feature extraction. 

## Sec IV. Algorithm Implementation and Development 

### Part (a) - Computing the Correlation Matrix for the first 100 Yale Faces from Dataset
For part a, I was tasked with computing the 100 x 100 correlation matrix for the first 100 images in the yalefaces dataset. We start by computing the correlation matrix C where we will compute the dot product (correlation) between the first 100 images in the matrix X. Each element of the correlation matrix is given by c_jk = x^T_j x_k where x_j is the jth column of the matrix X. We use NumPy's np.corrcoef function to compute the correlation matrix as the function calculates the correlation matrix using the correlation coefficient formula, which is normalized by the standard deviation of each variable. Since we want to compute the correlation between columns (i.e., each image column represents an observation), we set the rowvar parameter to False. 

```
X_corr = X[:, :100]  # Select first 100 columns
C = np.corrcoef(X_corr, rowvar=False)  # Compute correlation matrix
```
After computing the correlation matrix, we can plot it using the pcolor function from Matplotlib:

```
plt.pcolor(C)
plt.title("Correlation Matrix C")
plt.xlabel("Image")
plt.ylabel("Image")
plt.colorbar()
plt.show()

```
This produces a heatmap visualization of the correlation matrix, where each cell represents the correlation between two images on the scale of -1 to 1. 

### Part (b) - Determining Max and Min Correlation Faces from Subset

In this part of the project, I used the correlation matrix generated in part (a) to determine which two images in the Yale Faces dataset are most highly correlated and which are most uncorrelated. To do this, I first filled the diagonal of the correlation matrix with a negative value (-5 in this case) to avoid any interference with the correlation calculations. This was done using the ``` np.fill_diagonal()``` function set to -5.  

I then used the NumPy ```np.unravel_index()``` function to find the 2D index of the maximum and minimum correlation values in the matrix. For the maximum correlation images, I used ```np.unravel_index(C.argmax(), C.shape) ``` where ```C.argmax()``` finds the index of the maximum correlation value in the matrix C. The ```np.unravel_index()``` function then converts this index back to the original 2D index of C by using the original shape of C (C.shape) as a parameter. For the minimum correlation images, I used the same process except this time using ```np.abs(C).argmin()``` to convert the correlation values to absolute and then find the minimum value (value closest to 0). From these functions, the returned indices correspond to the two images that are most highly correlated and the two images that are most uncorrelated, respectively.

To visualize the pairs of images, I reshaped them back to their original size (32x32 pixels) and plotted them using matplotlib.pyplot.imshow(). I used plt.subplots_adjust() to adjust the horizontal spacing between the two subplots and plt.show() to display the plots. This is the code for plotting the most correlated images. Plotting the least correlated followed the same process.
```
reshaped_max1 = X[:,maxCorr[0]].reshape(32, 32)
# Plot the most highly correlated images
plt.subplot(1,2,1)
plt.imshow(reshaped_max1, cmap='gray')
plt.title('Most Highly Correlated Image ' + str(maxCorr[0]))
reshaped_max2 = X[:, maxCorr[1]].reshape(32, 32)
plt.subplot(1,2,2)
plt.imshow(reshaped_max2, cmap='gray')
plt.title('Most Highly Correlated Image ' + str(maxCorr[1]))

plt.subplots_adjust(wspace=0.6) # Adjust the horizontal spacing between subplots
plt.show()

```

### Part (c) - Computing a 10 x 10 Correlation Matrix for Specific Images
 
To compute the 10 x 10 correlation matrix for the specific images provided, I first extracted these images from the Yale Faces dataset and created a new matrix X2 with them. The specific image indexes given to me were: ```[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]```. I then subtracted 1 from each index to account for the 0-based indexing used in Python, through list comprehension: ```images = [image - 1 for image in images]```. I then extracted the images using ```X2 = X[:, images]``` where images contained the array of indexes.

Once I had the new matrix X2, I computed its correlation matrix C2 using the np.corrcoef function similarly to part **a)**. Once again, I specified rowvar = False to indicate that each column in the matrix X2 corresponds to an observation (i.e., an image).

Finally, I visualized the resulting correlation matrix C2 using a heat map created with ```plt.pcolor``` from the Matplotlib library, similarly to part **a)**. 

### Part (d) - Computing Eigenvectors of Y = XXT

To extract the principal components of the face images, I first compute the covariance matrix Y = XXT, where X is the matrix of images with each column representing an image. I can then find the eigenvectors of Y, which will correspond to the principal components of the images. The eigenvectors with the largest eigenvalues will capture the most variation in the data.

I first calculated the matrix Y using ```np.dot``` to multiply X with its transpose X.T:
```
Y = np.dot(X, X.T)
```
To then compute the eigenvectors and eigenvalues of Y, I used the ```np.linalg.eig()``` function from the numpy.linalg module. (LA = np.linalg)
```
eigenvalues, eigenvectors = LA.eig(Y)
```
Next, to extract the top six eigenvectors with the largest magnitude eigenvalues, I sorted the eigenvalues in descending order and sorted the eigenvectors by their corresponding eigenvalues:
```
index = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[index]
eigenvectors_sorted = eigenvectors[:, index]
top_six = eigenvectors_sorted[:, :6]
```
The resulting ```top_six``` matrix then contains the six eigenvectors with the largest magnitude eigenvalues. 

### Part (e) - Singular Value Decomposition (SVD) and Finding Principal Component Directions

To compute the principal component directions of the image dataset, I used the singular value decomposition (SVD) method on the yalefaces image matrix X. I did this by using the ```np.linalg.svd()``` function on the dataset X. (LA = np.linalg)
```
U, S, V = LA.svd(X)
```
The SVD method factorizes X into the product of three matrices: X = U*S*V.T, where U is an orthogonal matrix of left singular vectors, S is a diagonal matrix of singular values, and V.T is the transpose of an orthogonal matrix of right singular vectors.

To obtain the first six principal components, I then sliced U to the first six columns, corresponding to the six largest singular values. 
```
top_six = U[:, :6]
```
The resulting matrix contains the first six principal component directions as its columns. 

### Part (f) - Comparing First Eigenvector and First SVD Mode

In this section, I am tasked with comparing the first eigenvector obtained from the matrix Y in part **(d)** with the first SVD mode obtained from the matrix X in part **(e)** and then computing the norm of the difference of their absolute values.

To do this, I first extracted the eigenvector, v1, from the top six eigenvectors obtained in part **(d)**, and the first SVD mode, u1, from the SVD decomposition of the matrix X obtained in part **(e)**. 
```
v1 = top_six[:, 0]
u1 = U[:, 0]
```
Then, I computed the norm of the difference of the absolute values of v1 and u1 using the ```np.linalg.norm()``` function provided by the numpy library. (LA = np.linalg)
```
norm_difference = LA.norm(np.abs(v1) - np.abs(u1))
```
The resulting norm value provides a measure of the similarity between the first eigenvector and the first SVD mode. A smaller value indicates a higher degree of similarity, while a larger value indicates a greater difference between them.

### Part (g) - Computing Variance and Plotting SVD Modes

In this section, I am tasked with computing the percentage of variance captured by each of the first 6 SVD modes and to plot these modes.

To begin, I first store the first 6 modes in the variable ```first_six``` with ```first_six = U[:, :6]```. I then computed the percentage of variance captured by each mode using the formula ```np.cumsum(first_six**2)/np.sum(first_six**2)``` which calculates the cumulative sum of the squares of the elements in each of the first six SVD modes, and then divides by the sum of the squares of all elements in all SVD modes. The formula then returns an array with the 6 mode variances which I multiplied by 100 to get their percentage variances. 
 
I then plotted the first 6 SVD modes using the ```plt.imshow()``` function from the matplotlib library.

## Sec. V. Computational Results

### Part (a)

In part **a)**, I was tasked with computing a 100 × 100 correlation matrix C from the first 100 images in the dataset. The graph obtained by plotting the correlation matrix C is shown below:

![Screenshot 2023-04-18 at 10 35 43 PM](https://user-images.githubusercontent.com/122940974/232976433-66ae25b3-c3ae-4344-a2be-fe068ff76a5c.png)

By examining the plot, we can see that there are regions of the matrix where the correlation values are higher than in other regions. This is evident by the heatmap showing a brighter green/ yellow color. We can also see regions where there are negative correlations between images which imply that the images are almost inverses of each other. There are also regions where images have lower to almost no correlation with each other. Evident through the turquoise regions. From the plot, there is a diagonal line where the image pairs have correlations equal to 1. This is because these images are being paired up with themselves on this diagonal line and therefore their correlation value would be perfect. 

### Part (b)

For part **b)**, I was tasked with finding which two images were most highly correlated and which were most uncorrelated. The output of the code from part **b)** returns the two images. The two images that are most correlated are Images: 5 and 62. The two images that are least correlated are Images: 36 and 5. The images are displayed below:

![Screenshot 2023-04-18 at 10 36 42 PM](https://user-images.githubusercontent.com/122940974/232976579-7576c83c-27a2-4e67-93e1-5e4fabfd5e00.png)

The plots show that the two most highly correlated images are very similar, almost identical in fact, whereas the two most uncorrelated images are quite different with the significant amount of shade involved in the lighting condition of image 36. 

### Part (c) 

For part **c)**, I was tasked with computing the 10 × 10 correlation matrix between given images and plotting their correlation matrix. The graph obtained by plotting the new correlation matrix C2 is shown below:

![Screenshot 2023-04-18 at 10 37 07 PM](https://user-images.githubusercontent.com/122940974/232976639-05057bbe-c7d4-4060-8c4f-5cd141eef03c.png)

Much like in our plot from part **a)**. We can see regions of high correlations, low correlations and inverse correlations from the different color regions of the graph. Since the correlation matrix was created using only 10 different images from the dataset, it is a lot easier to spot and interpret the correlations between images as we can now clearly see each square correlation block on the graph. Much like **a)**, from the range of correlation values, there is still the same diagonal perfect correlation line on the graph where images are paired up with themselves and therefore perfectly correlated. 

### Part (d)

For part **d)**, I was tasked with creating the matrix Y = XXT and finding the first six eigenvectors with the largest magnitude eigenvalue. From the code calculations, I received the following eigenvectors with the largest eigenvalues:

![Screenshot 2023-04-18 at 10 37 29 PM](https://user-images.githubusercontent.com/122940974/232976704-5eeceb60-04ca-486a-8768-243f23c79dc5.png)

These eigenvectors show the directions of the maximum variance of the data. 

### Part (e)

For part **e)**, I was tasked with applying SVD on the matrix X and finding the first six principal component directions. From the SVD of the matrix X, the first six principal component directions are shown below:

![Screenshot 2023-04-18 at 10 38 18 PM](https://user-images.githubusercontent.com/122940974/232976829-e91ca847-1da9-4a1b-84df-b3812e30a647.png)

These vectors show the directions in the feature space that capture the most variance in the data.

### Part (f)
For part **f)**, I was tasked with comparing the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and computing the norm of difference of their absolute values. The results of the norm calculation are shown below:
```
The norm of the difference of the absolute values of the first eigenvector and the first SVD mode is:  6.087445984638146e-16
```
This shows that there is almost no difference between the first SVD mode and the first eigenvector. 

### Part (g)

In part **g)**, I was tasked with computing the percentage of variance captured by each of the first 6 SVD modes and plotting the first 6 SVD modes. The percentage variance captured by each of the 6 modes is shown below:
```
The percentage of variance captured by each of the first 6 SVD modes respectively is:  [0.00947502 0.04375777 0.09702215 0.12990517 0.1489301  0.15705217]
```
And the graphs obtained by plotting the first 6 SVD modes are shown below:

![Screenshot 2023-04-18 at 10 38 50 PM](https://user-images.githubusercontent.com/122940974/232976903-36559667-6f28-40f1-9a63-a134090a3ed6.png)


The first 6 SVD modes show the most dominant patterns in the data. We can see the major patterns and features that contribute to the overall variation in the data. For the yalefaces dataset, the first SVD mode may represent the lighting and contrast variations across the faces, while following modes may capture variations in facial expressions along with other features that represent the faces. 

## Sec. VI. Summary and Conclusion

In this project, I explored the Yale Faces dataset and performed various matrix computations and visualizations. I started by loading and visualizing the images in the dataset, followed by computing the correlation matrix for the first 100 images and visualizing it using a heat map. I then extracted the most highly correlated and least correlated images from the subset of images which I found were images (5, 62) and (36,5) respectively. From here, I then performed the same process on the first 10 images instead to get a better understanding and visualization of the correlations and feature relationships between images. Next, I found the first six eigenvectors with the largest eigenvalues and performed Singular Value Decomposition (SVD) on the dataset to also get the first six principal component direction vectors. I then compared the norm difference between the first principal component direction vector and the eigenvector and found the difference to be so small it could be considered negligible. Finally, I graphed the first six SVD modes and observed how these modes affect the reconstructed images. I saw that the first few modes contribute the most to the overall appearance of the faces, while the higher modes capture finer details.

In conclusion, this project provided an opportunity to explore matrix computations and their applications in image processing. The techniques I used, such as computing correlation matrices and performing SVD, are fundamental in many areas of data science and machine learning. Through this project, I gained insights into how these techniques can be used to analyze and manipulate image data.

