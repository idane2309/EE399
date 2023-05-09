# Homework 4: Feed Forward Neural Networks (Deep Learning)  - Ishan Dane

## Sec. I. Abstract:

In this project, I explored the application of feed-forward neural networks using PyTorch and SKlearn in two different scenarios. Firstly, I investigated the fitting of a three-layer feed-forward neural network to a given dataset consisting of X and Y values. The objective was to analyze the network's performance in capturing the underlying patterns and relationships within the data.

Next, I utilized the MNIST dataset, a well-known collection of handwritten digit images, to train a feed-forward neural network for classification purposes of the digit images. Prior to training, I performed Principal Component Analysis (PCA) to extract the essential features from the digit images. Subsequently, I then developed and tested a 3 layer neural network and compared the performance of the neural network against other classification algorithms such as LSTM, SVM, and decision trees.

Through this project, I gained valuable insights into the application of feed-forward neural networks in both regression and classification tasks. By fitting a three-layer neural network to a given dataset, I learned how to capture complex relationships and patterns within the data. Additionally, by training a feed-forward neural network on the MNIST dataset, I acquired hands-on experience in creating a deep learning classification model using real life datasets and evaluating its performance against other classification algorithms. This project enhanced my understanding of neural networks, their capabilities, and their potential for solving real-world problems.

## Sec. II. Introduction and Overview:

As a passionate machine learning enthusiast, this project allowed me to delve into the realm of neural networks, a subfield of machine learning inspired by the intricate functioning of the human brain. In this report, I present my exploration of feed-forward neural networks in two distinct scenarios: regression and classification.

The report is divided into two main sections, each addressing a specific application of feed-forward neural networks. In the first section, I develop and investigate the fitting of a three-layer feed-forward neural network to a given dataset consisting of X and Y values and optimize it using Adam Optimization algorithm. The objective is to analyze the network's capability to capture intricate patterns and relationships within the regression data.

Shifting focus to the second section, I explore the widely studied MNIST dataset, a benchmark for digit recognition tasks. Here, I delve into the application of feed-forward neural networks for digit classification. Before training the neural network, I employ Principal Component Analysis (PCA) to extract essential features from the digit images. This dimensionality reduction technique enhances the neural network's ability to classify digits accurately by representing the images more efficiently. I then train a three-layer-neural network to classify each digit to a high level of accuracy by employing Stochastic Gradient Descent (SGD) to optimize the model’s parameters. 

To provide a comprehensive evaluation, I then compare the performance of the feed-forward neural network against alternative classifiers such as Long Short-Term Memory (LSTM), Support Vector Machines (SVM), and decision trees. This comparative analysis offers insights into the strengths and weaknesses of different approaches to digit classification.

## Sec. III. Theoretical Background

Feed-forward neural networks serve as a foundational concept in artificial intelligence and are inspired by the workings of the human brain. As a machine learning engineer, it is crucial to comprehend the theoretical underpinnings of these networks, as they form the basis of many modern machine learning models.

A feed-forward neural network comprises interconnected layers of artificial neurons or perceptrons. These neurons receive input signals, apply mathematical transformations using activation functions, and produce output signals. The network's weights, which determine the strength of connections, are adjusted during training using the backpropagation algorithm.
In this project, I implemented the feed-forward neural networks using the PyTorch library, which provides a flexible and efficient framework for building and training neural networks. PyTorch simplifies the process of defining network architectures, applying activation functions, optimizing weights through backpropagation, and incorporating regularization techniques.

Feed-forward neural networks excel in regression tasks, capturing complex nonlinear relationships between input variables and continuous target variables. In classification tasks, these networks can accurately predict class labels by employing appropriate activation functions, such as softmax, and interpreting the output as probabilities across multiple classes.

By understanding the theoretical foundations of feed-forward neural networks and leveraging PyTorch, I gained the necessary expertise to design, implement, and optimize these models effectively in this project.





## Sec IV. Algorithm Implementation and Development 

## Section I. Given X, Y data
For section one of the assignment, I was tasked with working with a given dataset of X and Y values. Here is how I initialized the data:
```
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
### Part (i): Fit the data to a three layer feed-forward Neural Network
For this part, I implemented a three-layer feed-forward neural network using PyTorch. The network architecture is defined in the Net class, which is a subclass of the nn.Module base class provided by PyTorch. This allows me to leverage the functionality and utilities provided by PyTorch for building neural networks.

The Net class consists of three fully connected (linear) layers, denoted as fc1, fc2, and fc3. These layers are responsible for performing the computations and transformations within the network. The input size of the first layer is 1, as I am working with a 1-dimensional input. However, this can be adjusted based on the dimensionality of the input data in different applications.

In the ```__init__()``` method of the Net class, I initialize the layers of the neural network. For each layer, I specify the input size and the output size. For example, ```self.fc1 = nn.Linear(1, 10)``` defines the first linear layer with an input size of 1 and an output size of 10. Similarly, ```self.fc2``` and ```self.fc3``` have an input size of 10 and an output size of 10 and 1, respectively.The forward method is where the actual forward pass computations of the network are defined. In this method, I define the sequence of operations that the input data goes through to produce the output.

First, the input data x is passed through the first linear layer (fc1) using the ReLU activation function (torch.relu). The ReLU activation function introduces non-linearity into the network and helps in capturing complex patterns and relationships in the data.

The output of the first layer is then passed through the second linear layer (fc2) with ReLU activation, and finally through the third linear layer (fc3) without any activation function. This last layer provides the final output of the network.

By implementing the forward method, I have defined the flow of data through the neural network, enabling me to use it for forward propagation during training and inference. The code for my neural network is shown below:

```
#Feed forward 3 layer neural network
class Net(nn.Module):

   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(1, 10)
       self.fc2 = nn.Linear(10, 10)
       self.fc3 = nn.Linear(10, 1)
  
   def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = torch.relu(self.fc2(x))
       x = self.fc3(x)
       return x
```

### Part (ii): Using the first 20 data points as training data and last 10 points as test data, fit the neural network. Compute the least-square error for each of these over the training points and test data. 

For this part, the first step was to split the data and prepare it to be used in training the neural net. To prepare the data for training and testing the neural network, I performed several steps. Firstly, I divided the input data into training and testing subsets. For the training data, I selected the first 20 data points from the original dataset. I then reshaped and converted them into torch tensors. I then did the same for the corresponding target labels, both are shown below:

```torch.tensor(X[:20].reshape(-1, 1).astype(np.float32))``` for the input features (X_train) ```torch.tensor(Y[:20].reshape(-1, 1).astype(np.float32))``` for the corresponding target labels (Y_train).

Similarly, for the test data, I selected the remaining 10 data points from both original datasets and reshaped and converted them into torch tensors.

To facilitate the training and testing processes, I created separate datasets for the training and test data using ```torch.utils.data.TensorDataset```. The training dataset (train_dataset) consists of the input features (X_train) and target labels (Y_train), while the test dataset (test_dataset) consists of the input features (X_test) and target labels (Y_test).

I also utilized data loaders, ```train_loader``` and ```test_loader```, to efficiently load and iterate over the training and test datasets, respectively. The data loaders were created using ```torch.utils.data.DataLoader```, where I passed in the corresponding datasets (```train_dataset``` and ```test_dataset```). Additionally, I set the batch size to 1, meaning that each training and test sample is processed individually. I also ensured that the training data is shuffled (shuffle=True) during training, while the test data is not shuffled (shuffle=False) during evaluation.


 The network is then initialized using the Net class previously created, and I defined the criterion for the loss function as the Mean Squared Error (MSE) using ```nn.MSELoss()```. The optimizer used for updating the network parameters is Stochastic Gradient Descent (SGD) with a learning rate of 0.01.

```
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```

I set the number of epochs to 100, which represents the number of times the entire training dataset is passed through the network during training. For each epoch, I iterate over the training data using the ```train_loader```. Within this loop, I perform the forward pass of the data through the network using ```outputs = net(X_train_tensor)```. Then, I calculate the MSE loss between the predicted outputs and the actual labels using ```loss = criterion(outputs, Y_train_tensor)```. Where criterion was the ```nn.MSELoss()``` previously defined. 

After the forward pass, I proceed with the backward pass and optimization steps. I zero the gradients of the optimizer using ```optimizer.zero_grad()``, calculate the gradients of the loss with respect to the network parameters using ```loss.backward()```, and update the parameters using ```optimizer.step()```. This process of updating the parameters based on the gradients helps the network learn and improve its performance.

During training, I also print the loss after every 5 iterations to monitor the progress of the training process. This helps me analyze how the loss is decreasing over time and ensure that the network is converging.

```
num_epochs = 100

for epoch in range(num_epochs):
   for i, (X_train_tensor, Y_train_tensor) in enumerate(train_loader):
       # Forward pass
       outputs = net(X_train_tensor)
       loss = criterion(outputs, Y_train_tensor)
      
       # Backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
          
       if (i+1) % 5 == 0:
           print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

After training, I move on to testing the neural network on the remaining 10 data points, which serve as the test data. I use the ```test_loader``` to iterate over the test data. With the trained network, I pass the test data through it and calculate the MSE loss between the predicted outputs and the actual labels using ```mse_loss = criterion(outputs, Y_test_tensor)```. I accumulate the total loss over all the test data points as shown in the code below:
```
with torch.no_grad():
   total = 0
   for X_test_tensor, Y_test_tensor in test_loader:
       outputs = net(X_test_tensor)
       mse_loss = criterion(outputs, Y_test_tensor)
       total += mse_loss.item()
```

Finally, I calculate the average MSE loss on the test data by dividing the total loss by the number of batches in the test loader (len(test_loader)). This provides an evaluation metric to assess the performance of the trained network on unseen data.
```
print('MSE Loss on test data: {}'.format(total/len(test_loader)))
```

### Part (iii): Repeat (iii) but use the first 10 and last 10 data points as training data and the leftover points as test data. 

In this section, I repeated the process described in question (iii) by using a different set of training and test data. I used the first 10 and last 10 data points as the training data, while the middle 10 data points served as the test data.

To prepare the data, I performed the following steps similarly to the previous question. Firstly, I created the training dataset by concatenating the first 10 data points (X[:10]) with the last 10 data points (X[20:]) using ```np.concatenate()```. Similarly, I concatenated the corresponding target labels, Y[:10] and Y[20:]. I then reshaped and converted these concatenated arrays into torch tensors using ```torch.tensor(np.concatenate((X[:10], X[20:])).reshape(-1, 1).astype(np.float32))``` for the input features (X2_train) and ```torch.tensor(np.concatenate((Y[:10], Y[20:])).reshape(-1, 1).astype(np.float32))``` for the corresponding target labels (Y2_train).

For the test data, I selected the middle 10 data points (X[10:20]) and the corresponding target labels (Y[10:20]). Similarly, I reshaped and converted them into torch tensors using ```torch.tensor(X[10:20].reshape(-1, 1).astype(np.float32))``` for the input features (X2_test) and ```torch.tensor(Y[10:20].reshape(-1, 1).astype(np.float32))``` for the corresponding target labels (Y2_test).

To facilitate the training and testing processes, I once again created separate datasets for the training and test data using ```torch.utils.data.TensorDataset```. I then also utilized data loaders, ```train_loader2``` and ```test_loader2``` through ```torch.utils.data.DataLoader```, to efficiently load and iterate over the training and test datasets, respectively, as I did previously. 

I then initialized the new network using the Net class as like before, and defined the criterion for the loss function as the Mean Squared Error (MSE) using ```nn.MSELoss()```. The optimizer used for updating the network parameters is Stochastic Gradient Descent (SGD) with a learning rate of 0.01 once again. 

I kept the number of epochs at 100 for consistency in later evaluations and replicated the training loop using ```train_loader2``` with forward propagation and backward propagation as I did in the previous question. 

Finally, to test the neural network on the remaining 10 data points, I use ```test_loader2``` to iterate over the test data and calculate the MSE loss between the predicted outputs and the actual labels. I then calculate the average MSE loss on the test data by dividing the total loss by the number of batches in the test loader (len(test_loader)) like previously.

### Part (iv): Compare the models fit in homework one to the neural networks in (ii) and (iii)

** Part (iv) is a analysis and evaluation of the models, included in Section: Computational Results **

## Section II. MNIST Data

In the second section of my assignment, I work with the MNIST dataset of digit images and train a neural network to classify the digit images. 

To begin, I loaded the MNIST dataset using the ```fetch_openml()``` function from the ```sklearn.datasets``` module. I obtained 10,000 samples of handwritten digits, with each digit represented as a 28x28 grayscale image. The pixel values of the images were normalized to the range [0, 1] by dividing them by 255.0. The input images were stored in the X variable, and the corresponding labels were stored in the Y variable.
```
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data[:10000] / 255.0 # 10000 samples, 784 features
Y = mnist.target[:10000] # 10000 Labels
```

### Part (i): Compute the first 20 PCA modes of the digit images.
For the first part, I computed the first 20 PCA modes of the data. To compute the first 20 PCA modes of the digit images, I used the PCA class from the ```sklearn.decomposition``` module. I initialized an instance of PCA with the parameter ```n_components=20```, indicating that I want to extract the first 20 principal components. I then fitted the PCA model to the input images X using the ```.fit()``` method. This process calculated the principal components, which are orthogonal vectors that capture the maximum amount of variance in the data. I also used ```X_pca = pca.transform(X) ``` to apply the dimensionality reduction technique of Principal Component Analysis (PCA) to the input data X. The pca.transform(X) method takes the input data matrix X and projects it onto the principal components learned during the fitting process. This was done in case I needed a reduced X_train size for my neural network, in which I could use ```X_pca```. 

```
pca = PCA(n_components=20)
pca.fit(X)
X_pca = pca.transform(X)
```

To visualize the first 20 PCA modes, I plotted them using matplotlib. I created a figure with a 4x5 grid of subplots, representing 20 images. In each subplot, I used the ```.imshow()``` function to display the reshaped principal component (which is a 1D vector) as a grayscale image. I set the ```cmap``` parameter to 'gray' to display the images in grayscale. I then displayed the plot using ```plt.show()```.

By computing and visualizing the first 20 PCA modes, I gained insight into the most significant patterns and variations in the MNIST digit images. This analysis serves as a valuable preprocessing step for understanding the dataset and can potentially aid in feature extraction and dimensionality reduction for further machine learning tasks.

### Part(ii): Build a feed-forward neural network to classify the digits. Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers

In this part, I created a feed-forward neural net to perform digit classification on the digit images from the MNIST dataset. The dataset was preprocessed by loading it using the ```datasets.MNIST``` class from the **PyTorch** library. The dataset was split into separate training and testing sets. The training set was further processed using data loaders (```torch.utils.data.DataLoader```) , as done for the previous neural nets, to handle batching and shuffling during training and testing phases.

The neural network architecture, defined within the ```Net2()``` class, consisted of three fully connected (linear) layers. The input layer accepted flattened images of size 784, representing the 28x28 pixel grid. ReLU activation functions were applied after the first two hidden layers to introduce non-linearity. The output layer, consisting of 10 units, represented the probabilities for each digit class. The neural net class Net2 is shown below:
```
# 3 layer Feed Forward Neural Network to classify digits
class Net2(nn.Module):
   def __init__(self):
       super(Net2, self).__init__()
       self.fc1 = nn.Linear(784, 128)
       self.fc2 = nn.Linear(128, 64)
       self.fc3 = nn.Linear(64, 10)

   def forward(self, x):
       x = x.view(-1, 784) # Flatten the data (n, 1, 28, 28) --> (n, 784)
       x = torch.relu(self.fc1(x))
       x = torch.relu(self.fc2(x))
       x = self.fc3(x)
       return x
```

To train the model, the network was optimized using stochastic gradient descent (```torch.optim.SGD```). The chosen loss function was cross-entropy loss (```nn.CrossEntropyLoss```), appropriate for multi-class classification tasks. The optimizer iteratively adjusted the network's parameters to minimize the loss. The network was trained for a specified number of 10 epochs, with each epoch consisting of iterations over the training dataset. Intermediate training progress was reported, displaying the epoch number, step, and loss value. Much like the previous neural nets, a loop was used with forward and backward propagation to train the model. The code is shown below:
```
# Initialize the network and define the loss function and optimizer
net = Net2()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Train the network
num_epochs = 10
for epoch in range(num_epochs):
   for i, (images, labels) in enumerate(train_loader):
       optimizer.zero_grad()
       outputs = net(images)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()
      
       if (i+1) % 100 == 0:
           print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
```

After training, the network was tested on the separate testing dataset through the same setup as the previous networks. To evaluate the accuracy of the network's predictions, the number of correct predictions was recorded and compared to the total number of test samples. The overall accuracy of the network was then reported as a percentage. The test code is shown below:
```
# Test the network
with torch.no_grad():
   correct = 0
   total = 0
   for images, labels in test_loader:
       outputs = net(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()
      
   print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
```

To compare the accuracy of the neural network model against LSTM, SVM and decision tree classifiers, I had to create the classifiers and test them on the same MNIST data. Luckily, for both SVM and Decision Tree, I had already trained classifiers for the MNIST data on my previous assignment project. Therefore I was able to extract the code and results for SVM and Decision Tree. 

For the LSTM classification model, I had to create a new LSTM neural network. To perform digit classification using LSTM (Long Short-Term Memory) on the MNIST dataset, a recurrent neural network architecture was implemented. The code begins by configuring the device to utilize a CUDA-enabled GPU if available, or fallback to the CPU.

The LSTM network, defined within the LSTMNet class, consists of an LSTM layer followed by a fully connected layer. The LSTMNet class has been defined with the necessary parameters for initialization. The LSTM layer in the network is created using nn.LSTM with the specified input size, hidden size, and number of layers. The batch_first=True argument indicates that the input tensor has dimensions (batch_size, sequence_length, input_size). After the LSTM layer, a fully connected layer (nn.Linear) maps the hidden state to the number of classes (10 in this case).

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the LSTM network
class LSTMNet(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers, num_classes):
       super(LSTMNet, self).__init__()
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
       self.fc = nn.Linear(hidden_size, num_classes)
      
   def forward(self, x):
       h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
       c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      
       out, _ = self.lstm(x, (h0, c0))
       out = self.fc(out[:, -1, :])  # Extract the last output step and pass it through the fully connected layer
       return out
```

During training, the LSTM model is initialized and moved to the appropriate device, as determined by torch.device. The loss function used is the cross-entropy loss (nn.CrossEntropyLoss()), which is well-suited for multi-class classification tasks. The optimizer chosen is stochastic gradient descent (SGD) (torch.optim.SGD), which updates the network parameters to minimize the loss. Hyperparameters such as input size, hidden size, number of layers, number of classes, batch size, number of epochs, and learning rate have been defined.

The training loop iterates over the training data loader (train_loader). Each batch of images and labels is retrieved, and the images are reshaped to match the LSTM input dimensions using images.reshape(-1, 28, 28). The reshaped images and labels are then moved to the appropriate device. In the forward pass, the LSTM model processes the sequential input data and generates predictions. The loss is computed by comparing the predicted outputs with the true labels using the defined loss function. The optimizer is used to update the model parameters through backpropagation, following the steps of zeroing the gradients (optimizer.zero_grad()), performing backpropagation (loss.backward()), and applying the optimization step (optimizer.step()). Intermediate training progress is reported, indicating the current epoch, step, and loss value.

```
# Define hyperparameters
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64
num_epochs = 10
learning_rate = 0.01

# Initialize the LSTM model
model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
   for i, (images, labels) in enumerate(train_loader):
       images = images.reshape(-1, 28, 28).to(device)
       labels = labels.to(device)
      
       # Forward pass
       outputs = model(images)
       loss = criterion(outputs, labels)
      
       # Backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
      
       if (i+1) % 100 == 0:
           print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
```


After training, the model is switched to evaluation mode using model.eval(). Then, the accuracy is evaluated on the test set. The test set is iterated over using the test data loader (test_loader). Images from the test loader are reshaped and moved to the device, similar to the training process. Predictions are made using the trained LSTM model (model(images)), and the predicted labels are obtained by taking the index of the maximum value using torch.max(outputs.data, 1). The number of correct predictions is accumulated, and the total accuracy on the test set is calculated as a percentage of correct predictions.

```
# Evaluation on test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
   correct = 0
   total = 0
   for images, labels in test_loader:
       images = images.reshape(-1, 28, 28).to(device)
       labels = labels.to(device)
      
       outputs = model(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()
  
   accuracy = 100 * correct / total
   print(f"Accuracy on the test set: {accuracy:.2f}%.")
```

 
## Sec. V. Computational Results

## Section I. 

### Part (ii)
For this part, I created a neural network and fit the model to the first 20 points of a given X and Y dataset and tested the model on the remaining 10 points. The test results of the model showed a ```MSE Loss on test data: 186.97961148348722``` which is significantly high. This can be explained by the fact that the dataset the neural net was trained on (X) was very small. Due to the small size of the dataset, the neural network would have had a higher tendency to memorize the training examples instead of learning generalized patterns, consequently, performing poorly on the test set. 

Moreover, the training process would have exhibited higher variance in model performance. Due to the limited amount of data, the network's accuracy and loss values would have fluctuated significantly across different training runs. This instability can be attributed to the randomness in the data distribution and the network's initial weight initialization.

### Part (iii)
For this part, I created a neural network and fit the model to the first and last 10 points of a given X and Y dataset and tested the model on the remaining 10 points. The test results of the model showed a ```MSE Loss on test data: 8.99846659898758```. Relative to the previous part, the MSE loss this time was a lot smaller. This could be due to the training data provided to the network covering more variance and distribution of the points. Consequently the network would be more optimized to perform better on predictions as a clearer pattern can be seen. However, the MSE loss is far from low and is still relatively high. This would also be the result of using a small training dataset and the reasons previously mentioned.

### Part (iv)
In Homework 1, for a training set of the first 20 points from X and Y, and a test set of the remaining 10 points, the test least-squares error was:
```
Test Line fit error: 3.36363873604787
Test Parabola fit error: 8.713651781874919
Test Polynomial fit error: 28617752784.428474
```
Compared to the neural network’s mean-squared error of **186.98**, we can see that both the line fit and the parabola fit performed a lot better than the neural network by a magnitude of 10 in this regression case. The neural network still managed to outperform the polynomial fit but that is not saying much since the polynomial fit was purposely created to be an extreme overfitting model. 

For a training set of the first and last 10 points from X and Y, and a test set of the remaining 10 points, the test least-squares error from Homework 1 was:
```
Test Line fit error: 2.8065076975181618
Test Parabola fit error: 2.774982896893291
Test Polynomial fit error: 483.9099124568562
```
Compared to the neural network’s mean-squared error of **8.99%**, once again both the line fit and parabola fit outperformed the neural network but to a much smaller magnitude than the previous training set scenario. This further emphasizes the importance of selecting neural networks only when given a large dataset, else other machine learning algorithms may be more effective. 

## Section II. 
### Part (i)
For this part, I was asked to compute the first 20 PCA modes of the digit images from the MNIST dataset. The first 20 are shown below:


![Screenshot 2023-05-08 at 11 47 46 PM](https://user-images.githubusercontent.com/122940974/237016375-5d89ff7d-1629-42f2-9fa7-316ff53065f4.png)


### Part (ii)
For this part, I built a neural network to classify the digits in the MNIST dataset. The test results for my model showed an accuracy of **93.85 %** on the test data. The accuracy of my SVM model is **97.64%**, the accuracy of the Decision Tree on the data was **87.04% ** and the accuracy of the LSTM network on the MNIST data was **23.28%**. We can therefore see that the LSTM classifier had the worst accuracy out of all the algorithms in this case. 
 

## Sec. VI. Summary and Conclusion

In this assignment, I conducted a series of experiments to explore and evaluate different machine learning algorithms on various datasets. My objective was to gain insights into the performance and characteristics of these algorithms in different scenarios. Here is a comprehensive summary of my findings and the conclusions I drew from the experiments:

I performed PCA analysis on a dataset of digit images. By extracting the first 20 principal components, I visualized the captured variation and gained a compressed representation of the data. This analysis provided me with valuable insights into the most significant features of the dataset.

I trained 3-layer feed-forward neural networks on given X, Y data and the MNIST dataset. The neural network demonstrated its ability to learn and accurately classify digit images but did not do so well when it came to the regression task with the smaller dataset X, Y. I noticed that the limited size of the dataset and overfitting challenges posed limitations on the network's generalization capabilities.

I employed an LSTM (Long Short-Term Memory) classifier to classify the MNIST digit images. The LSTM model was the worst out of all the different algorithms. It achieved an accuracy of **23.28%** on the test set. We can therefore see that the LSTM classifier had the worst accuracy out of all the algorithms in this case. 

Additionally, I compared the performance of the neural network models with other classification algorithms, including SVM (Support Vector Machines) and decision trees. The neural network models did not always outperform these traditional algorithms, especially in the case of smaller datasets. 

Overall, I learned the importance of selecting appropriate algorithms based on the nature of the data and task at hand. The significance of dataset size and representativeness in determining model performance was evident. Further research and experimentation with larger datasets and advanced techniques can lead to enhanced performance and deeper insights in the field of machine learning.

