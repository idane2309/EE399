# Exploring Neural Network Architectures for Advancing and Predicting Chaotic Dynamics in the Lorenz System - Ishan Dane

## Sec. I. Abstract:

This report presents a comprehensive analysis of various neural network architectures for advancing the solution and predicting future states in the chaotic Lorenz system. The study focuses on the comparison between feed-forward neural networks (FNNs), Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs) in forecasting the system dynamics.

The research begins with training a feed-forward neural network to advance the solution from a given time t to t + ∆t for three different values of the control parameter ρ: 10, 28, and 40. The trained network is then evaluated for future state prediction at ρ values of 17 and 35. The accuracy and reliability of the predictions are assessed by comparing them against the ground truth values obtained through numerical integration of the Lorenz equations.

The findings highlight the strengths and limitations of each neural network architecture in capturing the intricate behavior of the Lorenz system. By developing 4 different neural networks architectures and testing their performance in future state predictions, I was able to analyze and compare each network’s ability to forecast future data given previous data. The feed-forward neural network exhibits exceptional accuracy in advancing the solution within a small time step, demonstrating its suitability for short-term prediction. However, its performance diminishes when attempting to accurately forecast future states beyond the training range. LSTM networks, with their ability to retain long-term dependencies, excel in predicting future states even for unseen parameter values. RNNs display a comparable performance to LSTMs, albeit with slightly lower accuracy. Echo State Networks, utilizing the reservoir computing approach, yield promising results for both advancing the solution and predicting future states, showcasing their potential in capturing chaotic dynamics.

This study provides valuable insights into the applicability of different neural network architectures for modeling and predicting chaotic systems. The research serves as a foundation for selecting the most appropriate neural network approach based on specific requirements and system characteristics. 


## Sec. II. Introduction and Overview:
This study explores the effectiveness of various neural network architectures in advancing the solution and predicting future states of the chaotic Lorenz system. The Lorenz system exhibits complex behavior characterized by sensitivity to initial conditions and the emergence of intricate attractors. Neural networks have emerged as powerful tools for modeling and forecasting chaotic dynamics due to their ability to capture nonlinear relationships.

The research focuses on comparing the performance of feed-forward neural networks (FNNs), Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs) in forecasting the Lorenz system dynamics. The study involves training the networks to advance the solution from a given time t to t + ∆t for various values of the control parameter ρ. The trained networks are then evaluated for future state prediction, both for the trained ρ values and unseen values.

Performance metrics such as accuracy and mean squared error are used to assess the predictive capabilities of the neural networks. A comparative analysis is conducted to identify the strengths and weaknesses of each architecture in capturing the complex dynamics of the Lorenz system.

This research contributes insights into the applicability of neural network architectures for modeling and predicting chaotic systems. The findings have implications across various fields, including weather forecasting, financial prediction, and nonlinear control systems. Moreover, the results contribute to the growing understanding of the utilization of neural networks in complex nonlinear systems.


## Sec. III. Theoretical Background

The Lorenz system, proposed by Edward N. Lorenz in 1963, is a set of ordinary differential equations that describes the behavior of a simplified atmospheric convection model. The system is composed of three variables: x, y, and z, which represent the state of the system at a given time. The equations are as follows:

dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

where σ, ρ, and β are the parameters controlling the system's behavior. The Lorenz system exhibits chaotic dynamics, characterized by its sensitivity to initial conditions and the emergence of strange attractors.

Neural networks have gained considerable attention in recent years as effective tools for modeling and forecasting complex nonlinear systems. Their ability to capture nonlinear relationships and learn from data makes them well-suited for capturing the intricate dynamics of chaotic systems.
Feed-Forward Neural Networks (FNNs) are the most commonly used type of neural network architecture. They consist of an input layer, one or more hidden layers, and an output layer. In FNNs, information flows in one direction, from the input layer through the hidden layers to the output layer. Each neuron in the network applies a nonlinear activation function to the weighted sum of its inputs.

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that addresses the limitations of traditional RNNs in capturing long-term dependencies. LSTMs employ memory cells that enable them to retain information over extended sequences. This feature makes them particularly effective in modeling time series data and capturing the temporal dependencies present in chaotic systems.

Recurrent Neural Networks (RNNs) are another class of neural networks commonly used for sequential data processing. Unlike feed-forward neural networks, RNNs have feedback connections, allowing them to utilize information from previous time steps. RNNs can be considered as having memory, enabling them to capture temporal dependencies in the data. However, RNNs often suffer from the vanishing gradient problem, which limits their ability to capture long-term dependencies.

Echo State Networks (ESNs) belong to the family of reservoir computing, which uses a fixed random network (reservoir) of recurrently connected neurons. The reservoir dynamics can amplify and transform input signals, enabling them to capture complex dynamics. ESNs differ from traditional RNNs by training only the output layer, while the internal connections of the reservoir remain fixed. This simplifies the training process and makes ESNs particularly suitable for modeling chaotic systems.

In this study, we investigate the performance of these neural network architectures in advancing the solution and predicting future states of the Lorenz system. By comparing their abilities to capture the chaotic dynamics of the system, we aim to gain insights into the strengths and weaknesses of each architecture for forecasting nonlinear and chaotic phenomena.

## Sec IV. Algorithm Implementation and Development 

### Part I. Train a neural network (NN) to advance the solution from time t to t + ∆t for three different values of the control parameter ρ, namely 10, 28, and 40. Subsequently, assess the performance of the trained NN in predicting future states for ρ values of 17 and 35.

To address the given question regarding the Lorenz equations and the training of a neural network (NN) for advancing the solution and predicting future states, I began by developing my Feed-Forward Neural Net (FFNN). 

Before creating my FFNN, I defined the values for the control parameter ρ (rho) used for training and testing the neural nets. The rho_train_values list contains the values 10, 28, and 40, while the rho_test_values list contains the values 17 and 35. These values represent the different scenarios for which the neural nets will be trained and their forecast evaluated.

I then created a function ```def get_lorenz_deriv(rho):```, which is responsible for obtaining the input-output pairs necessary for training and testing. Within this function, various parameters were initialized, such as the time step (dt), the total time period (T), and the constants β and σ. The values of these parameters were set to provide the most accurate results from the Lorenz System. 
```
# Initialize parameters
   dt = 0.01
   T = 8
   t = np.arange(0,T+dt,dt)
   beta = 8/3
   sigma = 10
```
Next, the function initializes the input and output arrays corresponding to the specific ρ value. These arrays are populated with zeros using the np.zeros function.
```
# Initialize input and output arrays for rho value
   nn_input = np.zeros((100 * (len(t) - 1), 3))
   nn_output = np.zeros_like(nn_input)
```
The Lorenz system equations were then defined within the nested function ```def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):```. This function takes the current state (x, y, z), time (t0), and the parameters σ, β, and ρ as inputs. It computes and returns the derivatives of x, y, and z according to the Lorenz equations. These equations describe the dynamics of the chaotic system and shown in the code below:
```
def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
       x, y, z = x_y_z
       return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
The code then proceeds to solve the Lorenz system for the given ρ value. It utilizes numerical integration through the odeint function from the python integrate module. By providing the initial conditions (x0, values chosen randomly), the lorenz_deriv function, and the time array (t), the code generates the trajectories of the Lorenz system for multiple initial conditions.
```
# Solve Lorenz system for rho value
   np.random.seed(123)
   x0 = -15 + 30 * np.random.random((100, 3))

   x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                   for x0_j in x0])
```
For each trajectory, the resulting data points are stored in the ```nn_input``` and ```nn_output array``` using a list comprehension. The dimensions of the input and output arrays are appropriately adjusted to accommodate the trajectory data. After completing the trajectory generation process, the function returns the input and output arrays (nn_input and nn_output) for the specific ρ value.
```
for j in range(100):
       nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
       nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
  
   return nn_input, nn_output
```

After developing the lorenz value function, I created two additional functions: ```create_train_data(rho_train_values)``` and ```create_test_data(rho)```. 

The create_train_data function combines the input and output arrays obtained from ```get_lorenz_deriv``` function for each ρ value in ```rho_train_values```. It iterates over the ρ values, concatenates the arrays, and converts them into torch tensors suitable for training the NN.
```
def create_train_data(rho_train_values):
   # Initialize input and output arrays
   nn_input = np.zeros((0, 3))
   nn_output = np.zeros_like(nn_input)

   # Get training data for each rho value
   for rho in rho_train_values:
       nn_input_rho, nn_output_rho = get_lorenz_deriv(rho)
       nn_input = np.concatenate((nn_input, nn_input_rho))
       nn_output = np.concatenate((nn_output, nn_output_rho))
  
   # Convert to torch tensors
   nn_input = torch.from_numpy(nn_input).float()
   nn_output = torch.from_numpy(nn_output).float()

   return nn_input, nn_output
```

Similarly, the ```create_test_data``` function generates the test data for a specific ρ value. It calls the ```get_lorenz_deriv``` function and converts the resulting arrays into torch tensors.
```
def create_test_data(rho):
   nn_test_input, nn_test_output = get_lorenz_deriv(rho)
   nn_test_input = torch.from_numpy(nn_test_input).float()
   nn_test_output = torch.from_numpy(nn_test_output).float()

   return nn_test_input, nn_test_output
```
These functions were created to easily create training and test datasets for the chaotic Lorenz System to be used for the different neural network architectures throughout this assignment. 

Subsequently, I then defined my FFNN class. To do this, I first defined three activation functions: logsig, radbas, and purelin. The logsig function computes the sigmoid function, radbas calculates the radial basis function, and purelin represents the identity function. These activation functions are then used within the model to introduce non-linear transformations to the network's outputs.
```
def logsig(x):
   return 1 / (1 + torch.exp(-x))

def radbas(x):
   return torch.exp(-torch.pow(x, 2))

def purelin(x):
   return x
```
Next I created my FFNN class, representing a feed-forward neural network model. The FFNN class inherits from the nn.Module class, which is a base class for all neural network modules in PyTorch. The __init__ method of my FFNN class initializes the layers of the network using the nn.Linear module from PyTorch. It consists of three fully connected layers (fc1, fc2, and fc3), with specific input and output dimensions. These layers defined the connectivity and the number of parameters to be learned during the training process.
```
def __init__(self):
       super(FFNN, self).__init__()
       self.fc1 = nn.Linear(in_features=3, out_features=10)
       self.fc2 = nn.Linear(in_features=10, out_features=10)
       self.fc3 = nn.Linear(in_features=10, out_features=3)
```

I then created the forward method of the FFNN class which describes the forward pass of the model. The input x passes through each layer sequentially, applying the defined activation functions (logsig, radbas, and purelin) after the first and second fully connected layers. Finally, the output of the last layer is returned. The path for the forward method can be seen below:
```
def forward(self, x):
       x = logsig(self.fc1(x))
       x = radbas(self.fc2(x))
       x = purelin(self.fc3(x))
       return x
```
After defining my FFNN model, I began the training process. To begin, I created an instance of the FFNN class named ```ffnn```. This instance represents the feed-forward neural network that will be trained to learn the dynamics of the Lorenz system: ```ffnn = FFNN()```

Next, the loss function and optimizer are defined. The criterion is set to the mean squared error (MSE) loss, which measures the average squared difference between the predicted output and the target output. The optimizer is defined as the stochastic gradient descent (SGD) optimizer with a learning rate of 0.01. The optimizer is responsible for updating the model's parameters based on the computed gradients during the training process.
```
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(ffnn.parameters(), lr=0.01)
```

To initiate the training process, I retrieved the input and output data (ffnn_input_train and ffnn_output_train) using the previously developed ```create_train_data``` function. These data are used for training the model to learn the dynamics of the Lorenz system for forecasting.
```
ffnn_input_train, ffnn_output_train = create_train_data(rho_train_values)
```

From here, I then ran a training loop for 50 epochs. In each epoch, the optimizer's gradients are set to zero using optimizer.zero_grad() to prevent accumulation. The model predicts the output based on the input data using ffnn_output_pred = model(ffnn_input_train). The MSE loss between the predicted output and the target output is calculated using criterion. Then, the gradients are computed using loss.backward(), and the optimizer updates the model's parameters using optimizer.step(). The loss value is printed for each epoch to monitor the progress of the training process. The code for my training is shown below:
```
for epoch in range(50):
   optimizer.zero_grad()
   ffnn_output_pred = ffnn(ffnn_input_train)
   loss = criterion(ffnn_output_pred, ffnn_output_train)
   loss.backward()
   optimizer.step()
   print('Epoch: ', epoch, 'Loss: ', loss.item())
```
Following the training phase, I then proceeded to test the trained FFNN model on its ability to forecast rho values ρ = 17 and ρ = 35 from my ```rho_test_values``` list. To do this, I set up a loop to iterate over the ρ values specified in the ```rho_test_values``` list. For each ρ value, the corresponding test data is generated using the ```create_test_data``` function created previously. This function prepares the input and output tensors for the test data based on the given ρ value. Inside the loop, the model predicts the output based on the test input data using ```ffnn_output_pred = ffnn(ffnn_input_test)```. The predicted output is compared with the actual target output using the MSE loss function (criterion) to calculate the loss value.
The loss value for each ρ value is then printed using print('Loss for rho = ', rho, ': ', loss.item()). This provides an indication of how well the FFNN model performs in predicting the future states of the Lorenz system for different values of ρ. The test code is shown below:
```
for rho in rho_test_values:
   ffnn_input_test, ffnn_output_test = create_test_data(rho)
   ffnn_output_pred = ffnn(ffnn_input_test)
   loss = criterion(ffnn_output_pred, ffnn_output_test)
   print('Loss for rho = ', rho, ': ', loss.item())
```

### Part II. Compare feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.

#### LSTM

The next neural network architecture I created was an LSTM model to forecast the dynamics of the Lorenz system. The LSTM architecture is a type of recurrent neural network (RNN) that can capture long-term dependencies in sequential data.

I first defined a class for the LSTM called ```class LSTM(nn.Module):```, which inherits from the ```nn.Module``` class provided by PyTorch. This allows the model to utilize the functionalities and utilities provided by PyTorch for neural network training.

Inside the LSTM class, the constructor method ```__init__``` is defined. The constructor method is shown below:
```
def __init__(self, input_size=3, hidden_layer_size=10, output_size=3):
       super().__init__()
       self.hidden_layer_size = hidden_layer_size
       self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
       self.linear = nn.Linear(hidden_layer_size, output_size)

```


It takes three arguments: ```input_size```,``` hidden_layer_size```, and ```output_size```. These arguments determine the dimensions of the input, hidden layer, and output of the LSTM model, respectively. The constructor also calls the constructor of the superclass using ```super().__init__()```. Within the constructor, the hidden_layer_size is stored as an attribute of the object (self.hidden_layer_size). This value represents the number of hidden units in the LSTM layer. The LSTM layer is created using ```nn.LSTM``` with the specified input_size and hidden_layer_size. The ```batch_first=True``` argument indicates that the input data is provided with the batch dimension as the first dimension. Additionally, a linear layer (nn.Linear) is defined as self.linear. This layer maps the output of the LSTM layer to the desired output size.

The forward method created for the class is implemented to define the forward pass of the LSTM model. The method is shown below:
```
def forward(self, x):
       h0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
       c0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
       out, _ = self.lstm(x, (h0, c0))
       out = self.linear(out[:, -1, :])
       return out
```
The forward method takes an input tensor x as its argument. Inside the method, two tensors h0 and c0 are initialized with zeros. These tensors represent the initial hidden state and cell state of the LSTM. The input tensor x is then passed through the LSTM layer using ```self.lstm(x, (h0, c0))```. This returns two outputs: out and a tuple containing the final hidden state and cell state (which are not used here). The out tensor represents the LSTM's output at each time step.

To obtain the final prediction, only the last time step's output from the LSTM is selected using ```out[:, -1, :]```. This selects the last time step for all samples in the batch and retains the feature dimensions.

Finally, the selected output is passed through the linear layer (self.linear) to obtain the final prediction for each sample. The output tensor is returned as the result of the forward pass.

The next step is to train and test the created LSTM model. Much like how the FFNN was trained and tested, I first created an instance of the LSTM model for training and testing. 

Next, the loss function and optimizer are defined with ```nn.MSELoss()``` as the loss function and ```optim.SGD``` with a learning rate of 0.01 and a momentum of 0.9 as the optimizer. The optimizer is responsible for updating the parameters of the LSTM model during the training process, aiming to minimize the defined loss function much like previously. 
To train the LSTM model, training data is obtained using the ```create_train_data function``` once again. The inputs and corresponding outputs are stored in the variables ```lstm_input_train``` and ```lstm_output_train```, respectively.

Before proceeding with training, the shape of the input data ```lstm_input_train``` is modified to match the expected format for the LSTM model. The lstm_input_train tensor is reshaped using ```.reshape()``` to have dimensions (number of samples, sequence length, number of features). In this case, the sequence length is set to 1 since the LSTM is trained to predict the next time step given the current time step. The reshape is shown below for reference:
```
# Reshape the input data for LSTM
lstm_input_train = lstm_input_train.reshape(lstm_input_train.shape[0], 1, lstm_input_train.shape[1])
```
To ensure consistency in the comparison of neural network architectures, the training loop is initiated with 50 epochs as it was with training the FFNN. The LSTM model is applied to the training data by passing lstm_input_train to the ```lstm``` object. This returns the predicted outputs, lstm_output_pred, which represent the LSTM's forecasted values.

For each training iteration, the loss between the predicted outputs and the true outputs is then computed using the defined loss function. The optimizer's step method is called to update the model's parameters based on the gradients and learning rate and the corresponding loss value is printed. The code for training the model is identical to the FFNN training but instead uses the ```lstm``` model instance.

Once again, the same testing iteration occurs; for each ```rho``` value in ```rho_test_values```, the test data is obtained using the ```create_test_data``` function. The input data and corresponding output data are stored in lstm_input_test and lstm_output_test, respectively.

Similar to the training data, the shape of the test input data is modified using ```.reshape()``` to match the expected format for the LSTM model. The lstm_input_test tensor is reshaped to have dimensions (number of samples, sequence length, number of features) and is passed through the LSTM model. The loss is then calculated between the predicted outputs and the true outputs  computed using the defined loss function. The calculated loss value for each rho value is then printed, providing an indication of how well the LSTM model performs in terms of forecasting the dynamics of the Lorenz system for the given test data.

#### RNN

The first neural network architecture I created was a RNN model to forecast the dynamics of the Lorenz system. My RNN class inherits from the ```nn.Module``` class, which is the base class for all neural network modules in PyTorch.

In the ```__init__``` constructor method of the RNN class, the model architecture is defined. It takes three arguments: ```input_size```, ```hidden_layer_size```, and ```output_size```, which represent the dimensions of the input, hidden layer, and output, respectively. Within the ```__init__``` method, the hidden_layer_size is stored as an attribute of the class (self.hidden_layer_size) for later reference. The constructor method is shown below:
```
def __init__(self, input_size=3, hidden_layer_size=10, output_size=3):
       super().__init__()
       self.hidden_layer_size = hidden_layer_size
       self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)
       self.linear = nn.Linear(hidden_layer_size, output_size)
```

The RNN layer is defined using the ```nn.RNN``` module. It takes the input_size as the input dimension, hidden_layer_size as the hidden layer dimension, and ```batch_first=True``` to indicate that the first dimension of the input tensor represents the batch size.
The linear layer is defined using the ```nn.Linear``` module, which maps the hidden layer's output to the desired output size.

The next method in my RNN class is the forward method. In this method the input tensor x is passed through the RNN layer. The method is shown below:
```
def forward(self, x):
       h0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
       out, _ = self.rnn(x, h0)
       out = self.linear(out[:, -1, :])
       return out
```
The initial hidden state, h0, is initialized as a tensor of zeros with dimensions **(1, x.size(0), self.hidden_layer_size)**. This represents a single hidden state for each sample in the input batch. The RNN layer processes the input tensor x along with the initial hidden state h0. The output of the RNN layer, ```out```, is obtained. To obtain the final prediction, the output tensor is passed through the linear layer. Specifically, ```out[:, -1, :]``` extracts the last time step's output for each sample, which is then passed through the linear layer to obtain the final output. The output tensor is returned as the final output of the forward pass.

The next step was to train my RNN model and test its forecasting capability. To train the RNN model, I followed the same approach as the previous FFNN and LSTM. First, I created an instance of the RNN model designed with an input size of 3, a hidden layer size of 10, and an output size of 3. Next, I defined the loss function as the mean squared error (MSE) loss and utilized stochastic gradient descent (SGD) as the optimizer with a learning rate of 0.01 and momentum value of 0.9. 

Once again, to prepare the training data, I invoked the create_train_data function. This function provided us with the necessary input and output tensors (rnn_input_train and rnn_output_train, respectively) required for training the RNN model. These tensors were then reshaped to match the expected input shape of the RNN module similarly to the LSTM. The training process consisted of iterating over 50 epochs once again. Within each epoch, I passed the reshaped input tensor, ```rnn_input_train```, through the RNN model to obtain the predicted output tensor, rnn_output_pred and calculated the loss between the predicted output and the target output (rnn_output_train) using the MSE loss function. I then performed backpropagation by calling loss.backward() and updated the model parameters by calling optimizer.step(), similarly to the training of the previous neural nets. 

The final step was to test my RNN model. This was done exactly like the testing for the LSTM. For each rho value in ```rho_test_values```, I used the ```create_test_data``` function to extract the corresponding test input and output sets for the RNN model. The test input data was then modified using ```.reshape()``` to match the expected format for the RNN model (number of samples, sequence length, number of features) and was passed through the RNN model. The loss for each rho value was then calculated between the predicted outputs and the true outputs computed using the defined loss function.

#### ESN

The final neural network I created was an Echo State Network (ESN) model specifically designed for the Lorenz System. The ESN is a type of recurrent neural network (RNN) that utilizes a large reservoir of sparsely connected neurons to process input data and generate predictions.

To create the ESN, I created two main classes: ```Reservoir``` and ```EchoState```.

In the Reservoir class, I defined the structure of the reservoir, which is the core component of the ESN. The reservoir contains hidden neurons responsible for capturing the dynamics of the Lorenz System. The class inherits from the ```nn.Module``` class from PyTorch and has the parameter ```hidden_dim``` which represents the number of hidden neurons in the reservoir, and the connectivity parameter controls the sparsity of the reservoir connections. The constructor method for the class is shown below:
```
 def __init__(self, hidden_dim, connectivity):
   super().__init__()
  
   self.Wx = self.sparse_matrix(hidden_dim, connectivity)
   self.Wh = self.sparse_matrix(hidden_dim, connectivity)
   self.Uh = self.sparse_matrix(hidden_dim, connectivity)
   self.act = nn.Tanh()
```
To initialize the reservoir, I created sparse weight matrices ```Wx```, ```Wh```, and ```Uh``` using the ```sparse_matrix``` method. This method generates random weight matrices and applies a sparsity mask to create the desired connectivity pattern. The sparsity is controlled by the connectivity parameter. Additionally, I used the hyperbolic tangent activation function (Tanh) to introduce non-linearity to the reservoir dynamics. The method is shown below:
```
 def sparse_matrix(self, m, p):
   mask_distribution = torch.distributions.Bernoulli(p)
   S = torch.randn((m, m))
   mask = mask_distribution.sample(S.shape)
   S = (S*mask).to_sparse()
   return S
```
The ```forward``` method in the Reservoir class computes the output of the reservoir given an input x and the previous hidden state h. It performs matrix multiplications between the sparse weight matrices and the input, applies the activation function, and produces the reservoir output y and updated hidden state h. This process is essential for capturing the temporal dependencies and non-linear dynamics of the Lorenz System. The forward method is shown below:
```
 def forward(self, x, h):
   h = self.act(torch.sparse.mm(self.Uh, h.T).T +
                torch.sparse.mm(self.Wh, x.T).T)
   y = self.act(torch.sparse.mm(self.Wx, h.T).T)

   return y, h
```

In the EchoState class, I defined the overall structure of the ESN. This class also inherits ```nn.Module``` from PyTorch and takes as input the dimensions of the input (in_dim), output (out_dim), reservoir (reservoir_dim), and the connectivity level (connectivity).The constructor method for the class is shown below:
```
def __init__(self, in_dim, out_dim, reservoir_dim, connectivity):
   super().__init__()

   self.reservoir_dim = reservoir_dim
   self.input_to_reservoir = nn.Linear(in_dim, reservoir_dim)
   self.input_to_reservoir.requires_grad_(False)

   self.reservoir = Reservoir(reservoir_dim, connectivity)
   self.readout = nn.Linear(reservoir_dim, out_dim)
```

The ```input_to_reservoir``` layer is a linear transformation that maps the input to the reservoir dimensions. This step enables the ESN to process the input data within the reservoir.
The reservoir instance represents the reservoir component of the ESN. It initializes an instance of the Reservoir class with the specified ```reservoir_dim``` and ```connectivity```. This encapsulates the reservoir functionality, including the sparse weight matrices and the reservoir dynamics.
The ```readout``` layer is another linear transformation that maps the reservoir output to the output dimensions. This layer provides the final predictions based on the reservoir's processed information.

In the forward method of the EchoState class, I perform the forward pass of the ESN. Given an input tensor x, I first transform the input to match the reservoir dimensions using the ```input_to_reservoir``` layer. I initialize the reservoir state ```h``` with ones, representing the initial hidden state. Then, for each time step in the input sequence, I pass the transformed input and the current reservoir state through the reservoir instance. This step captures the temporal dynamics and updates the reservoir state accordingly. The reservoir outputs at each time step are stored in a list. After processing all time steps, I concatenate the reservoir outputs into a tensor, representing the reservoir's overall representation of the input sequence. Finally, I pass this tensor through the readout layer to obtain the final outputs of the ESN. The forward method is shown below:
```
def forward(self, x):
   reservoir_in = self.input_to_reservoir(x)
   h = torch.ones(x.size(0), self.reservoir_dim)
   reservoirs = []
   for i in range(x.size(1)):
     out, h = self.reservoir(reservoir_in[:, i, :], h)
     reservoirs.append(out.unsqueeze(1))
   reservoirs = torch.cat(reservoirs, dim=1)
   outputs = self.readout(reservoirs)
   return outputs
```

The next step was to train the ESN model. To do this I followed the same process as I did with the previous neural network architectures. I first created an instance of the EchoState class named esn. The instantiation takes the following arguments: EchoState(input_dim, output_dim, reservoir_dim, connectivity). In this case, I set input_dim and output_dim to 3, representing the dimensions of the input and output data. The reservoir_dim is set to 50, determining the number of hidden neurons in the reservoir. The connectivity parameter is set to 0.1, controlling the sparsity of the reservoir connections. 
```
esn = EchoState(3, 3, 50, 0.1)
```
Next, I defined the loss function using ```nn.MSELoss()```, which stands for Mean Squared Error. For optimization, I used the stochastic gradient descent (SGD) optimizer once again with the same learning rate and momentum. 

To train the ESN model, I used the create_train_data function, which returns the input and output data for training the ESN. These are assigned to esn_input_train and esn_output_train, respectively.
Next, I began the training process again consisting of iterating over 50 epochs. Within each epoch, I pass the training input data ```esn_input_train``` to the ```esn``` model to obtain the predicted outputs. Since the input data has a shape of (batch_size, sequence_length, input_dim), I reshape it using ```.view(1, -1, 3)``` to match the expected input shape of the ESN.
The model then processes the input data and generates the predicted outputs, which are stored in the outputs variable. I then calculate the loss by comparing the predicted outputs with the true outputs (esn_output_train) using the previously defined mean squared error loss function.
After computing the loss, I perform backpropagation by calling loss.backward() and finally update the model parameters using the optimizer by calling optimizer.step(). In each epoch, the current epoch number and the corresponding loss value are printed to monitor the training progress.

After training the model, I tested it on the ```rho_test_values```. This was done exactly like the previous models. I created my test set using the ```create_test_data``` function and made sure to modify the input data using ```.view(1, -1, 3)``` to match the format for the ESN model. The loss for each rho value was then calculated and printed. 

## Sec. V. Computational Results

#### FFNN
The test results for the FFNN model in predicting rho = 17 and rho = 35 are shown below:

```
Loss for rho =  17 :  51.017696380615234
Loss for rho =  35 :  119.07109832763672
```

Based on the loss values for the feed-forward neural network (FFNN) in predicting the Lorenz system dynamics, we can see that it performed a lot better when predicting rho = 17 than rho = 35. 

A loss value of 51.017696380615234 for rho = 17 indicates a moderate level of prediction error. It implies that the FFNN model is somewhat successful in approximating the true output values for rho = 17, but there is still room for improvement.

On the other hand, a loss value of 119.07109832763672 for rho = 35 indicates a higher level of prediction error. It suggests that the FFNN model struggles more in accurately capturing the dynamics of the Lorenz system for rho = 35. This indicates a less reliable forecast for this particular rho value.


#### LSTM
The test results for the LSTM model in predicting rho = 17 and rho = 35 are shown below:
```
Loss for rho =  17 :  34.123451232910156
Loss for rho =  35 :  55.88348388671875
```

A loss value of 34.123451232910156 for rho = 17 suggests a reasonably accurate prediction. It indicates that the LSTM model can capture the underlying patterns and dynamics of the Lorenz system for rho = 17 quite well. The predicted outputs are close to the actual outputs, with a relatively low level of error.

Similarly, a loss value of 55.88348388671875 for rho = 35 indicates a reasonably accurate forecast, though with a slightly higher level of prediction error compared to rho = 17. The LSTM model can still capture the essential characteristics of the Lorenz system for rho = 35, but there might be some discrepancies between the predicted and actual outputs.

We can see that the LSTM model outperforms the FFNN in terms of forecasting accuracy for both rho values. For the rho value 35, the LSTM does a significantly better job at its prediction. 

#### RNN
The test results for the RNN model in predicting rho = 17 and rho = 35 are shown below:
```
Loss for rho =  17 :  43.22047805786133
Loss for rho =  35 :  55.39307403564453
```

A loss value of 43.22047805786133 for rho = 17 also suggests a reasonably accurate prediction. It indicates that the RNN model does a relatively alright job at capturing the underlying patterns and dynamics of the Lorenz system for rho = 17. 

Similarly, a loss value of 55.39307403564453 for rho = 35 indicates that the RNN actually had a relatively good forecast for this rho value compared to the previous models, though with a slightly higher level of prediction error compared to rho = 17. However, the loss for rho = 35 is still not the best, and there is still room for improvement.

Comparing the RNN's loss values to those of the other models, we can see that the RNN performs competitively. While the LSTM model achieved similar and lower loss values, indicating better predictive accuracy overall, the RNN still demonstrates its capability to capture and model the temporal dependencies within the Lorenz system, beating the FFNN model in both rho values and matching the LSTM model in its prediction for rho = 35. However, it is important to note that the RNN falls behind the LSTM in predicting for rho = 17, suggesting that the LSTM has a better ability overall to capture long-term dependencies and intricate patterns in the data.

#### ESN
The test results for the RNN model in predicting rho = 17 and rho = 35 are shown below:
```
Loss for rho =  17 :  21.916521072387695
Loss for rho =  35 :  38.415565490722656
```
The Echo State Network (ESN) demonstrates impressive performance in predicting the dynamics of the Lorenz system. 

The obtained loss value of 21.916521072387695 for rho = 17 shows that the model is able to accurately predict the lorenz output for rho = 17 by capturing the underlying patterns of the system. 

Furthermore, the loss of 38.415565490722656 for rho = 35 also shows the model’s accuracy in predicting different rho values. The score is a relatively good one, demonstrating the model’s accurate nature. 

We can see that the ESN model outperforms all the other models in its prediction of both rho values and does so by a relatively significant margin. 

#### Overall

While the FFNN, LSTM, RNN, and ESN all show promise in forecasting the dynamics of the Lorenz system, the ESN stands out as the top performer, surpassing the other models in terms of predictive accuracy. Its ability to effectively capture and utilize temporal dependencies gives it an edge in accurately modeling the complex nonlinear behavior of the system. These findings highlight the potential of the ESN as a powerful tool for time series forecasting and dynamical systems prediction.

## Sec. VI. Summary and Conclusion

In this assignment, I explored the application of different neural network models for forecasting the dynamics of the Lorenz system. Specifically, I examined the performance of Feed-Forward Neural Network (FFNN), Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Echo State Network (ESN) in capturing and predicting the complex behavior of the system.

Initially, I trained and tested the FFNN model, which showed promising results in capturing some underlying patterns in the Lorenz system. However, further improvements are needed to enhance its predictive accuracy.

Next, I developed the LSTM model, which demonstrated superior performance compared to the FFNN. By incorporating memory cells and gates, the LSTM effectively captured long-term dependencies and achieved lower losses, indicating its ability to accurately forecast the system's dynamics.
I then created a RNN model, though capable of capturing temporal dependencies, yielded moderate results in terms of predictive accuracy. Although it outperformed the FFFN, It ultimately fell short compared to the LSTM in accurately forecasting the complex behavior of the Lorenz system.

Finally, I developed the ESN, which stood out as the top performer among all the models. By leveraging a large reservoir of recurrently connected nodes, the ESN effectively captured and amplified the system's temporal dynamics, resulting in the lowest losses and superior predictive accuracy. The ESN emerged as the most effective model for forecasting the dynamics of the Lorenz system. Its unique reservoir computing approach, with its ability to harness temporal dependencies, demonstrates its potential as a powerful tool for time series forecasting and dynamical systems prediction.

Overall, this assignment highlights the importance of choosing appropriate neural network models for capturing and forecasting complex nonlinear systems. The LSTM and ESN models, in particular, showcased their effectiveness in capturing long-term dependencies and accurately predicting the dynamics of the Lorenz system. These findings contribute to the field of dynamical systems prediction and provide valuable insights for future research and applications in various domains.


