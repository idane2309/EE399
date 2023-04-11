# Homework 1: Machine Learning Least-Squares Error Method

## Sec. I. Abstract:

In this assignment, I implemented machine learning techniques to fit different models to a given dataset and evaluate the performance of the model. The dataset consists of 31 data points. The first part of the assignment involved finding the parameters of a given model that minimizes the least-squares error. To do this I employed a minimization function to determine the parameters from the error. In the second part, I generated 2D loss landscapes by plotting different combinations of fixing 2 parameters and sweeping the other 2 parameters. This allowed me to determine the number and location of minimums for the parameters being swept. 
In the third and fourth parts, I employed various models: a line fit, parabola fit, and high-degree polynomial fit. The models were fit to subsets of the dataset and evaluated on the remaining points. The results between them were compared to assess the model's generalization performance. 

## Sec. II. Introduction and Overview:


Machine learning provides a range of powerful tools for analyzing and understanding complex data and applying it in estimations through models created from the data. However, the origin of models play a significant role in their success  and therefore it is important to be able to evaluate the performance of different types of models through methods such as comparing their ‘least squared-error’. In this homework assignment, we were tasked with experimenting with the fundamentals of least-squared error on a real-world dataset. To do this, we want to fit evaluate the performance of different models on a dataset of 31 data points. 

The first model we are tasked to consider is a trigonometric function with linear and constant terms, which we aim to fit using the least-squares method. We first write code to find the parameters of the model that minimize its least-squares error through a minimize function provided by the Scipy library. With our optimized parameters, we then investigate the different 2D loss landscapes of the model by plotting the error grids of different combinations of two fixed and two swept model parameters and determining the location and number of minima.

In the third and fourth parts of the assignment, we examine the performance of different models on different subsets of the dataset as training and test data. First, we fit a line, parabola, and 19th degree polynomial to the first 20 data points chosen as training data. From this we get the model coefficient parameters that best optimize the different fittings. We then evaluate the models’ performance on the remaining 10 points of the dataset chosen as test data by comparing their least-squared errors. We then repeat this process using the first 10 and last 10 data points as training data and evaluating the models on the middle 10 points as test data.

Through these exercises, I have gained valuable insights into the challenges of model fitting and error analysis for decision making. I also gained an understanding for the importance of model selection and evaluation for real-world applications.

## Sec. III. Theoretical Background

In the field of machine learning, we try to model a system or phenomenon through analyzing its data. We do this by defining a function or set of them from the data that map inputs to outputs, and then optimizing the parameters of the function(s) to minimize the error difference between the predicted and actual outputs. Least-squared error is a commonly used approach for fitting models to data and evaluating their performance. When using this approach, the goal is to minimize the sum of the squared errors between the model predictions and the actual output data points. This can be achieved by adjusting the model parameters until the error is at its minimum point. 

It is important to consider how the complexity of a model plays a factor in the model’s least-squares error and its ultimate success. A model that is too simple (e.g a simple line fit) may not capture the underlying patterns and relationships of the data and therefore underfitting the data. On the other hand, a model that is too complex (e.g a super high degree polynomial) may overfit the data by also fitting in the noise instead of the underlying signal. This would result in it ‘memorizing’ the training data instead of learning the general pattern of it. Both overfitting and underfitting results in a model that performs poorly on new unseen data. To avoid overfitting or underfitting, we split the dataset into a training and test set and then evaluate the model on the test set.

In this assignment, we will use the least-squares error approach to fit and compare different models to a given dataset of 31 points. We will first fit a model using the function f(x) = A cos(Bx) + Cx + D. We will then explore and  identify the optimal parameters A, B, C, and D for the model that minimize the error. Additionally, we will explore the effects of model complexity by fitting a line, parabola, and 19th degree polynomial to the data and comparing their performance on both training and test data through their least-squares error. Through this exercise, we will gain insight on the trade-offs between model complexity and performance in machine learning.

## Sec. IV. Algorithm Implementation and Development 
For this assignment, we were given a dataset of 31 datapoints. 
```
X = np.arange(0,31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
### i)
In the first task, we were given a model defined as ```f(x) = A*cos(B*x) + C*x + D ``` and asked to fit it to the dataset using the least-squares error method. The goal was to determine the parameters A, B, C and D that would provide the minimum error. 
The first step was to define the error function(velfit). The function takes in an array of the four parameters (c), the input data (x), and the observed output data (y). It then calculates the error between the model’s (with parameters c) predictions and the actual observed output, and returns the root-mean squared error e2. 

RMS Formula:    <img width="200" alt="Screenshot 2023-04-10 at 12 03 16 AM" src="https://user-images.githubusercontent.com/122940974/230846477-601f3e42-871c-44d5-8709-3ee42491dd09.png">

```
def velfit (c, x, y):
    e2 = np.sqrt(np.sum((c[0] * np.cos(c[1] * x) + c[2] * x + c[3] - y)**2)/ len(y))
    return e2
```

To begin the optimization process, I set an initial guess for the parameters (v0) and passed it to the minimize function from the SciPy optimization library, that takes the objective function (velfit), the input and output data (X and Y) and the method as Nelder-Mead (a numerical method used to find the minimum of an objective function)
```
res = opt.minimize(velfit, v0, args=(X, Y), method='Nelder-Mead')
```
The optimization algorithm then finds the values of the parameters that minimize the objective function and returns the result as an optimization object (res). From here, I would then extract the optimized parameters through ``` c = res.x ```

With c now containing the optimized parameters, I graphed the model’s predictions against the training data. To do this, I created a new set of x values ranging from 0 to 31 with a step size of 0.01, and used the new optimized parameters (c) in ```f(x)``` to calculate the predicted y-values for each x-value.
```
x2 = np.arange(0, 31.01, 0.01)
yfit2 = (c[0] * np.cos(c[1] * x2) + c[2] * x2 + c[3])

```
From here, using the Matplotlib library, I plotted the training data as black dots on the graph, and the model predictions as a red line on the same graph.
```
plt.plot(np.arange(0, 31), Y, 'ko')
plt.plot(x2, yfit2, 'r-')
plt.show()
```
I then calculated the minimum error of the optimized model using the objective error function (velfit). 
```
min_error = velfit(c, X, Y)
```
### ii) 
With the results from **i)**, I performed a parameter sweep analysis by sweeping through a range of values for pairs of parameters while keeping the other pair fixed in their original values. These combinations would then be put into the error function (velfit) to generate and plot a 2D loss landscape. The sweep helps to visualize values of the parameters and locate minimums to evaluate the parameters’ impact on the error of the model. 

To do this, I first set the range of values that I wanted each parameter to sweep through, using the NumPy ```np.linspace() ``` function to create the grid of values. 
```
A_values = np.linspace(0, 20, 100)
B_values = np.linspace(0, 10 * np.pi/4, 100)
C_values = np.linspace(-10, 15.1, 100)
D_values = np.linspace(20, 50, 100)
```
I then assigned another group of variables to store the original optimized parameters from c.
```
A = c[0]
B = c[1]
C = c[2]
D = c[3]
```

For the first combination I fixed A and B and swept through values for C and D, I first created a 2D array (error) to hold the error values from calling the error function (velfit) for this combination.
```
error = np.zeros((len(C_values), len(D_values)))
```
Now to calculate the error point for each grid value in the sweep, I used a nested for loop iterating over the lengths of ```C_values``` and ```D_values``` . In the nested for loop, I would set the array of parameters being passed into the error function based on the sweep combination and then call the error function (velfit) to calculate the value for each error grid accordingly. 
```
for i in range(0, len(C_values)): 
    for j in range(0, len(D_values)):
        params = [A, B, C_values[i], D_values[j]]
        error[i, j] = velfit(params, X, Y)   
```
With the 2D error array filled out, I then plot the error values in a 2D loss landscape using a pseudocolor plot: 
``` plt.pcolor(C_values, D_values, error) ```

**This process would be repeated with new values for every possible combination with the 4 parameters to create 5 more graphs.**

### iii)
For the third task, I used the first 20 data points as training data and was required to fit a line, parabola and 19th degree polynomial to the data. Once the different models were created, the task required computing the least-square error for each of these over the training points. Then the final exercise was to compute the least square error of these models on the test data which were the remaining 11 data points from the dataset.

First, I created the training and testing data sets by splitting the X and Y arrays by array slicing. The first 20 datapoints were used as training data and the remaining 11 points were used as testing data. 
```
# Training Data
X_train = X[:20]
Y_train = Y[:20]
# Testing Data
X_test = X[20:]
Y_test = Y[20:]
```
I then fit three different polynomial curves to the training data using NumPy’s ```np.polyfit()``` function to create the 3 different models. The first line fit model represented a polynomial fit of degree of 1, the second parabola fit model represented a polynomial fit of degree of 2, the third high degree polynomial fit represented a polynomial fit of degree 19. By using NumPy’s polyfit function, I was returned with the most optimized coefficients representing the data for each fit. 
```
line_fit = np.polyfit(X_train, Y_train, 1)
parab_fit = np.polyfit(X_train, Y_train, 2)
poly_fit = np.polyfit(X_train, Y_train, 19)
```
I then calculated the RMS errors of each of the models by defining three separate loss functions: line_fit_loss, parab_fit_loss and poly_fit_loss. Each function takes in the polynomial coefficients calculated previously, the input data (X), and the output data (Y) as arguments and calculates the RMS errors of each model using the equation provided in **i)**. For the parab_fit_loss and poly_fit_loss functions, I took advantage of NumPy’s polyval function which evaluates a polynomial with a given array of coefficients and x. 
```
def line_fit_loss(c, x, y):
    line_e = np.sqrt(np.sum((c[0] * x + c[1] - y)**2) / len(y))
    return line_e

def parab_fit_loss(c, x, y):
    parab_e = np.sqrt(np.sum((np.polyval(c, x) - y)**2) / len(y))
    return parab_e

def poly_fit_loss(c, x, y):
    poly_e = np.sqrt(np.sum((np.polyval(c, x) - y)**2) / len(y))
    return poly_e
```
Consequently, I calculated the errors of each model on the training data using their respective loss functions. Then I calculated the errors of each model on the remaining test data using the same loss functions. This information can then be used to compare the performance of the different models to select the best fitting model.

### iv)
For the third task, I am repeating the process of **iii)** with the same 3 models (linear, quadratic and polynomial with degree 19) but this time using the first 10 and last 10 data points as training data and the remaining middle data points as testing data.

To split the data into training and testing sets, I used the same method of array slicing with concatenation. 
```
# Training Data
X_train = np.concatenate((X[:10], X[-10:]))
Y_train = np.concatenate((Y[:10], Y[-10:]))
# Test Data
X_test = X[10:21]
Y_test = Y[10:21]
```
With the training and test sets ready to go, I fitted the same three models to the training data using ``` np.polyfit(X_train, Y_train, degree) ``` to obtain the optimized fit coefficients. From here I computed the errors for the models on both the training and test data using the functions defined in **iii)** line_fit_loss, parab_fit_loss, poly_fit_loss. This code is tested how well the three models fit to the new training and test data.

## Sec. V. Computational Results

### i)
In the first task, I obtained the optimized coefficients for the model ```f(x) = A*cos(B*x) + C*x + D ``` using the minimize function from Scipy.  This gave me the following coefficients:
```
A = 2.1716818723637914
B = 0.9093249029166655
C = 0.7324784894461773
D = 31.45291849616531
# f(x) = 2.1716818723637914 cos (0.9093249029166655x)  + 0.7324784894461773x + 31.45291849616531
```

Next, graphing the optimized model with the actual observed output data Y showed the following results.

![Screenshot 2023-04-10 at 5 44 38 PM](https://user-images.githubusercontent.com/122940974/231026679-871d0e56-9ce1-4a3b-892f-e6a9f4cc8d37.png)

The resulting graph showed that the model predictions (red line) fit the training data (black dots) to an extent, especially at the start and end regions where the data points are more concentrated. However, there are some deviations between the model and the data towards the middle of the x-range. This suggests that the model may not be accurately capturing the underlying trend in those regions, and may need to be improved by incorporating additional features or adjusting the model structure.
The resulting minimum error from the model is then ```Min error: 1.592725853040056```

### ii)
In the second task, I performed parameter sweeps to explore the changing effects of different parameters on the error of the model from **i)**. I then plotted the generated error grids for each pair of swept parameters to create the 2D loss landscapes for evaluation. This showed the following results.

![Screenshot 2023-04-10 at 7 14 50 PM](https://user-images.githubusercontent.com/122940974/231042876-dac95577-290d-4e8f-b2e7-a1844d2394b5.png)


From this combination of fixing A, B and sweeping C,D we can see that there is a bundle of minimas around where the range of D = 30 to D = 35. This bundle of minimas on the graph starts at D = 34 and C = -10 and follows a slight decline down the D values as C values increase. 

![Screenshot 2023-04-10 at 7 15 03 PM](https://user-images.githubusercontent.com/122940974/231042998-d88e31e3-4c1d-40ae-999f-3cec43dde482.png)

From this combination of fixing C, D and sweeping A, B we can see 3 different bands of minimas at A = 2.5, A = 13.5 and A = 18 for all values of B. 

![Screenshot 2023-04-10 at 7 15 09 PM](https://user-images.githubusercontent.com/122940974/231043054-9d23a2d5-323b-40d4-b756-d5d9632cb463.png)

From this combination of fixing A, D and sweeping B, C we can see a clear band of minimas at around B = 3.3 for all values of C. 

![Screenshot 2023-04-10 at 7 15 21 PM](https://user-images.githubusercontent.com/122940974/231043091-6f144ced-1ed0-485e-99ec-45b8d5b03309.png)

From this combination of fixing B, C and sweeping A, D we can see a cluster of minimas at around A = 6 to A = 8.5 and D values smaller than D = 30. 

![Screenshot 2023-04-10 at 7 15 29 PM](https://user-images.githubusercontent.com/122940974/231043136-82078bf5-4a9b-440b-8a86-f6fa38f97e28.png)

From this combination of fixing A, C and sweeping B, D we can see a band of minimas at around D = 2.5 to D = 3.5 for all values of B. We also see a minima at B = 44 for values of D up to D = 3. 

![Screenshot 2023-04-10 at 7 15 35 PM](https://user-images.githubusercontent.com/122940974/231043183-9c30e32d-51f6-4f9b-922a-3a68520557ef.png)

From this combination of fixing B, D and sweeping A, C we can see a clear band of minimas at around A = 8.5 for all values of C. 

### iii)
For the third task, I created three different models: a line fit model, a parabola fit model and a 19th degree polynomial fit model. These models were trained on the first 20 data points and tested on the remaining data points. Then I computed the least-squares errors of these models on the training and test data. The results were the following:
```
Training Line fit error: 2.242749386808538
Training Parabola fit error: 2.1255393482773766
Training Polynomial fit error: 0.028351503968806435
Test Line fit error: 3.36363873604787
Test Parabola fit error: 8.713651781874919
Test Polynomial fit error: 28617752784.428474
```
From these results, we can see that the training error for the line fit and parabola fit models were low. The test error for these models were also on the lower end and were relatively similar to the training error. The parabola fit had a higher deviation between training and test error. However, when it came to the 19th degree polynomial model, the training error was incredibly low compared to the other models but then the test error was absurdly high. This was a clear indicator of overfitting as the model had “memorized” the data and had not learned its general pattern, resulting in significant deviations with predicting values that did not belong to the training set. The 19th degree polynomial model vs the actual observed output data can be seen below. 

![Screenshot 2023-04-10 at 8 37 08 PM](https://user-images.githubusercontent.com/122940974/231049873-255a1abf-7b4b-464c-811c-9038bd67abb5.png)

### iv)
For the fourth task, I deployed the same three models but trained them on the first 10 and last 10 data points and tested them on the remaining middle data points. Computer their least-squares errors on the training and test data gave me these results:
```
Training Line fit error: 1.851669904329375
Training Parabola fit error: 1.8508364115957907
Training Polynomial fit error: 0.1638133765080727
Test Line fit error: 2.8065076975181618
Test Parabola fit error: 2.774982896893291
Test Polynomial fit error: 483.9099124568562
```
From these results we can see a similar situation to **iii)**. Our line fit and parabola fit models have low error values with little deviation between the training error and test error. However the 19th degree polynomial fit model has a significantly lower training error but once again an significantly higher test error. As previously explained, this is a clear indication of overfitting. The plot of the 19th degree polynomial model vs the actual observed output data can be seen below.

![Screenshot 2023-04-10 at 8 37 44 PM](https://user-images.githubusercontent.com/122940974/231049941-1c0ef31c-72a5-4aa1-9139-19355838255f.png)

Between **iii)** and **iv)** we can see that the training errors for the line fit and parabola fit models using the training data from **iv)** resulted in a lower error compared to the model training errors in  **iii)**. This may be because the middle data points are more scattered compared to the first and last 10 points and therefore would have caused deviations in the models’ learning of the general pattern in **iii)**. However, the training error for the polynomial model was higher in **iv)** than in **iii)**. For the test errors, once again, the test errors for the models are lower in **iv)** than in **iii)** most likely for the same reasons, the models were able to better learn the underlying pattern of the data using the first and last 10 data points. Even the the polynomial model’s test error was much lower than it was in **iii)** by a significant amount. From the graphs, we can see that for the polynomial model, the significant error occurs when the model goes past the 20 data points it was trained on and begins to deviate significantly. However, in **iv)** we can see that most of the error from the polynomial model occurs in the middle section of the data points that it was not trained on. 

## Sec. VI. Summary and Conclusion

In this project, I optimized and graphed a given model of ```f(x) = A*cos(B*x) + C*x + D ``` on a dataset of 31 points, first using a minimize function from SciPy and then using parameter sweeps through different combinations of parameters to locate locations of parameters minimas. From this I found that just using the minimize function by itself may not be enough and should be incorporated with optimization methods such as parameter sweeping to further improve the model. Next, I explored and generated three different models: a line fit model, a parabola fit model, and a 19th degree polynomial fit model. I trained these models on different subsets of the data and computed their least-squares errors on both the training and test data. Here i found that the line fit and parabola fit models had relatively low errors and showed little deviation between training and test errors. However, the 19th degree polynomial fit model had a significantly lower training error but a much higher test error, indicating overfitting. This model was not able to learn the general pattern of the data but rather memorized the training set. I also found that training the models on the first 10 and last 10 data points provided a much lower error output on the test set than training the models on the first 20 data points. 

In conclusion, the choice of model and training data directly impacts how accurately the model can predict the output of the data set with relatively low error. We saw that the 19th degree polynomial fit model showed overfitting and was not able to generalize to the test data. Therefore, we can conclude that when working with data sets, it is important to identify when overfitting or underfitting is occurring, select the right model that fits the data, as well as to select the right subset of data to train and test the model. 
