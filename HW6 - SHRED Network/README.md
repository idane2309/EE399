# Exploring SHRED: Sensing and Decoding High-Dimensional Spatio-Temporal States using LSTM-Based Decoders 
# Ishan Dane

## Sec. I. Abstract:
In this code report, I explore the application of SHallow REcurrent Decoders (SHRED) in the context of sensing and decoding high-dimensional, spatio-temporal states. The goal is to understand and analyze the performance of the SHRED model in various scenarios using sea-surface temperature (SST) data. The report covers four key analyses: evaluating the model's performance as a function of the time lag variable, assessing its performance with the addition of Gaussian noise to the data, analyzing its performance as a function of the number of sensors, and training the model and visualizing the results.

To achieve these objectives, I utilized the code and data provided in the GitHub repository by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz and followed the provided iPython notebook example.ipynb for implementation and analysis.

First, I trained the SHRED model on the SST dataset and plotted the obtained results. By visualizing the predictions, I gained insights into the model's ability to capture the spatio-temporal patterns in the sea-surface temperature.

Next, I conducted an analysis of the model's performance as a function of the time lag variable. This involved modifying the code to experiment with different lag values and evaluating the model's performance using appropriate metrics. I analyzed the results to identify any trends or patterns that emerged as the time lag varied.

Furthermore, I investigated the impact of Gaussian noise on the model's performance. By adding Gaussian noise to the input data, I observed how the model's predictions were affected. I examined the performance metrics at different noise levels to assess the model's robustness and sensitivity to noise.

In addition, I explored the relationship between the number of sensors and the model's performance. By varying the number of sensors and evaluating the model's performance, I gained insights into the trade-off between the number of sensors and the accuracy of the reconstructed high-dimensional state.

Through these analyses, I gained a comprehensive understanding of the SHRED model's capabilities and limitations. The report presents detailed results, visualizations, and conclusions for each analysis, providing a comprehensive overview of the model's performance in different scenarios.


## Sec. II. Introduction and Overview:

The ability to accurately sense and decode high-dimensional, spatio-temporal states plays a crucial role in various scientific and engineering domains. Understanding the underlying patterns and dynamics within complex datasets is essential for making informed decisions and predictions. In this code report, I explore the application of SHallow REcurrent Decoders (SHRED) in the context of sensing and decoding high-dimensional states, with a particular focus on sea-surface temperature (SST) data.

The SHRED model, proposed by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz, offers a powerful framework for mapping trajectories of sensor measurements to high-dimensional, spatio-temporal states. By leveraging the capabilities of LSTM-based decoders, SHRED aims to capture the underlying dynamics and relationships within the data, enabling accurate reconstruction of the high-dimensional state.

This report revolves around several key analyses that aim to evaluate and understand the performance of the SHRED model. The primary objectives include:
Evaluating the model's performance as a function of the time lag variable: By varying the time lag between the sensor measurements and the corresponding state observations, we can explore how the model's accuracy and prediction capabilities are influenced by different temporal dependencies.
Assessing the model's performance with the addition of Gaussian noise to the data: Introducing noise to the input data allows us to investigate the robustness and resilience of the SHRED model. By analyzing its performance under varying noise levels, we can gain insights into its ability to handle noisy or imperfect sensor measurements.
Analyzing the model's performance as a function of the number of sensors: Exploring the relationship between the number of sensors and the accuracy of the reconstructed high-dimensional state provides valuable insights into the trade-off between sensing complexity and decoding accuracy.

To accomplish these objectives, I utilized the code and data provided in the GitHub repository for the paper "Sensing with shallow recurrent decoder networks." The repository offers a comprehensive example codebase and datasets, including the SST dataset, which serves as the primary focus of this analysis.


## Sec. III. Theoretical Background

**_SHallow REcurrent Decoders (SHRED) Model_**

The SHRED model is a powerful framework proposed by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz, which aims to map trajectories of sensor measurements to high-dimensional, spatio-temporal states. It leverages the capabilities of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to capture and model the temporal dependencies present in the data.

The SHRED model consists of an encoder-decoder architecture. The encoder takes as input a sequence of sensor measurements, representing the observed state over time. The encoder extracts and encodes the relevant features from the input sequence, capturing the underlying patterns and dynamics.

The decoder, based on LSTM units, takes the encoded representation from the encoder and reconstructs the high-dimensional, spatio-temporal state. The LSTM units maintain internal memory, allowing them to capture and utilize information from previous time steps. This characteristic makes LSTM-based decoders well-suited for modeling sequences with long-term dependencies.

During the training phase, the SHRED model optimizes the decoder's parameters to minimize the discrepancy between the reconstructed high-dimensional state and the ground truth state observations. This optimization is typically performed using techniques such as backpropagation and gradient descent.

**_Time Lag Variable_**

The time lag variable plays a significant role in the sensing and decoding of high-dimensional, spatio-temporal states. It represents the temporal relationship between the sensor measurements and the corresponding state observations. By considering different time lags, we can capture different temporal dependencies and explore their impact on the model's performance.

A longer time lag allows the model to capture slower dynamics and trends in the data, potentially improving the accuracy of the reconstructed state over longer time horizons. On the other hand, a shorter time lag emphasizes more immediate dependencies and may lead to better short-term predictions.
Analyzing the performance of the SHRED model as a function of the time lag variable provides insights into the optimal lag duration for capturing the relevant temporal patterns and achieving accurate reconstructions.

**_Noise Analysis_**

Real-world sensor measurements are often subject to various sources of noise, which can introduce uncertainties and affect the accuracy of the reconstructed state. Understanding the model's performance under noisy conditions is crucial for evaluating its robustness and reliability.
In the context of the SHRED model, adding Gaussian noise to the input data allows us to simulate noisy sensor measurements. By analyzing the model's performance under different levels of noise, we can examine its ability to filter out noise and still produce accurate state reconstructions. This analysis provides valuable insights into the model's robustness in practical scenarios where sensor measurements may be corrupted by noise.

**_Number of Sensors_**

The number of sensors used in the sensing and decoding process has an impact on the accuracy and efficiency of reconstructing high-dimensional, spatio-temporal states. Increasing the number of sensors provides more information and potentially enhances the model's ability to capture fine-grained details of the state. However, it also increases the complexity and cost of the sensing system.

Analyzing the performance of the SHRED model as a function of the number of sensors allows us to understand the trade-off between the number of sensors and the accuracy of the reconstructed state. By varying the number of sensors and evaluating the model's performance, we can determine the optimal sensor configuration that balances accuracy and resource constraints.


## Sec IV. Algorithm Implementation and Development 

### Training the SHRED model and Plotting the results
In order to train the SHRED model and plot the results, several steps were taken. Firstly, to define the model's input, three sensor locations were randomly selected. These locations represent the specific measurement points in the sea-surface temperature (SST) dataset. Additionally, the trajectory length (lags) was set to 52, which corresponds to one year of measurements.
```
num_sensors = 3
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```

Next, the dataset was divided into three subsets: training, validation, and test sets. Random indices were chosen for the training set, ensuring a diverse representation of the data. Next, the code creates a mask to separate the training set indices from the validation and test set indices, and generates the indices for the validation and test sets by filtering the mask. Consequently, ```valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]``` retrieves the indices in the mask array where the value is not 0. These indices correspond to the indices for the validation and test set which are selected from the valid_test_indices. The resulting arrays of indices can then be used to divide the data into the respective sets for training, validation, and testing purposes. This division allowed for comprehensive evaluation and analysis of the model's performance.
```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```

To prepare the data for training, the sklearn library's ```MinMaxScaler``` was employed. This scaler was used to preprocess the SST data by normalizing it within a specific range. By fitting the scaler to the training data, the scaling parameters were determined and applied to transform the entire dataset. This step ensured that all features were on a similar scale, optimizing the training process and enabling accurate comparisons.
```
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)
```
To train the SHRED model and prepare the data for training, I implemented the following steps. First, I generated the input sequences required for the SHRED model. I created a NumPy array called ```all_data_in``` with a shape of ```(n - lags, lags, num_sensors)```. This array serves as a container to store the input sequences that will be fed into the SHRED model.
```
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
   all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
```

Next, I proceeded to generate the training, validation, and test datasets utilizing their respective indices arrays for both the reconstruction of states and the forecasting of sensors. This involved several steps:
I determined the device to be used for training by checking if a CUDA-enabled GPU is available. If a GPU is available, I set the device to 'cuda'; otherwise, I set it to 'cpu'.
```device = 'cuda' if torch.cuda.is_available() else 'cpu'```
I created Torch tensors for the input sequences of the training, validation, and test datasets. These tensors were derived from the ```all_data_in``` array, using the indices obtained from the previous step. For example, train_data_in was generated by selecting the input sequences corresponding to the train_indices from all_data_in. The tensors were also cast to the torch.float32 data type and moved to the selected device.
```
train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
```
Similarly, I created Torch tensors for the output sequences of the training, validation, and test datasets. These tensors were derived from the transformed_X array, using the indices obtained by adding the lags - 1 offset to the respective indices. The tensors were cast to torch.float32 and moved to the selected device.
```
### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)
```
To facilitate the training process, I created custom datasets for the training, validation, and test data. These datasets were instantiated using the TimeSeriesDataset class, with the input and output tensors as arguments.
```
train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
By following these steps, I successfully prepared the input sequences and created the necessary datasets for training, validation, and testing of the SHRED model.

To train the SHRED model,  the model was first initialized using the ```models.SHRED``` class implemented in the ```models.py``` file, passing in the appropriate parameters. 
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
```
Next, I performed the training of the SHRED model and obtained the validation errors. This was achieved by calling the ```models.fit``` function from ```models.py```, and passing in the instantiated SHRED model, along with the training and validation datasets previously created. 
```
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```
Finally, to assess the performance of the SHRED model on the test dataset and compare it with the ground truth data, I utilized the following code snippet.
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
```
First, I applied the trained SHRED model to the test dataset by passing it through the model using ```shred(test_dataset.X)```. The resulting reconstructed sensor measurements were obtained using ```detach().cpu().numpy()``` to detach the tensor from the computation graph, move it to the CPU, and convert it to a NumPy array.

Since the sensor measurements were previously normalized using the MinMaxScaler, I used ```sc.inverse_transform``` to reverse the normalization and obtain the reconstructed sensor measurements in their original scale. This was done for both the reconstructed sensor measurements (test_recons) and the ground truth sensor measurements (test_ground_truth).

To evaluate the performance of the SHRED model, I then computed the normalized Euclidean distance between the reconstructed sensor measurements and the ground truth sensor measurements. This was achieved by calculating the Frobenius norm (```np.linalg.norm```) of the difference between test_recons and test_ground_truth, divided by the Frobenius norm of test_ground_truth. This normalization ensures that the distance metric is relative to the magnitude of the ground truth data. This value would then allow for an assessment of the accuracy of the SHRED model in capturing the underlying spatio-temporal patterns in the sea-surface temperature data. The results were then plotted to show the learning curve of the model and the ground truth vs reconstruction difference. 

### 1. Do an analysis of the performance as a function of the time lag variable
To analyze the performance with differing time lag variables, I defined a set of time lags to train the SHRED model on. These values are shown below:
```more_lags = [1, 2, 4, 8, 16, 32, 52]```
The same training process was utilized, however this time, for each ```lag``` in ```more_lags``` a new model was created, using its value, and its performance assessed with the error being appended to an ``error`` list for ultimate comparison. The error was then used to plot the relative error associated with the lag value. 

### 2. Do an analysis of the performance as a function of noise (add Gaussian noise to data)
To analyze the performance with added levels of noise added to the data, I defined a list of noise levels in the variable ```noise_levels```. These noise levels represent the standard deviation of the Gaussian noise that will be added to the data. The levels include 0.0 (no noise), 0.01, 0.05, 0.1, 0.2, and 0.3.

Next, a loop was executed to iterate through each ```noise level``` in the ```noise_levels list```. Within each iteration, I introduced noise to the data based on the noise level by adding Gaussian noise to the original data ```load_X``` shown below:
```noisy_X = load_X + np.random.normal(loc=0.0, scale=noise_level, size=(load_X.shape))```

A SHRED model was then trained for each ```noise_level``` and its performance assessed with the corresponding noise level. The model’s error was then appended to an ```error``` list which was then used to plot the relative error associated with the noise level. 

### 3. Do an analysis of the performance as a function of the number of sensors
To analyze the performance with differing number of sensors, I defined a list of sensor counts in the variable ```num_sensors_list```. The values in the list are shown below:
```num_sensors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]```

Once again, a loop was executed to iterate through each ```num_sensor``` value in the ```num_sensors_list```. Within each iteration, the ```sensor_loc``` variable is reassigned a random selection of sensor locations based on the current ```num_sensor``` value. The same training process is then repeated using the ```num_sensor``` value and the performance of the SHRED model is then assessed based on the current ```num_sensor``` value. Each iteration’s error is then appended to the ```error``` list which is then used to plot the relative error associated with the number of sensors. 

## Sec. V. Computational Results
### Training the SHRED model and Plotting the results
After training the initial SHRED model with parameters: ```num_sensors = 3, lags = 52, noise_level = 0```, the model was assessed by calculating the Frobenius norm of the difference between test_recons and test_ground_truth, divided by the Frobenius norm of test_ground_truth. The error calculated yielded a value of:
```Error: 0.019603658```

Trained on 1000 epochs, the learning curve of the model is shown below:

![Screenshot 2023-05-22 at 11 24 38 PM](https://github.com/idane2309/EE399/assets/122940974/8b7c1acc-f0ac-43ca-ad5b-31a6daf97c82)


From the graph we can see that the validation error drops significantly between 0 and 20 epochs, then begins to level out after 20 epochs, slowly decreasing down from there. 

To show the difference between the ground truth and reconstruction values, the plot of them alongside each other is shown below:


![Screenshot 2023-05-22 at 11 24 57 PM](https://github.com/idane2309/EE399/assets/122940974/4ba9d7f4-6454-4e59-bd9c-facc4b49c31b)


The graph of the ground truth against the reconstruction values shows a strong similarity. We can see that the reconstruction values is able to accurately capture the trends in the in comparison to the ground truth values, as well as the sudden changes and spike patterns from the sensors.


### 1. Do an analysis of the performance as a function of the time lag variable
The graph of the relative error associated with differing time lag variables is shown below:


![Screenshot 2023-05-22 at 11 25 19 PM](https://github.com/idane2309/EE399/assets/122940974/961270d5-6db6-47b3-892f-25da02dd5039)


From the graph we can see that overall, as the time lag increases, the relative error decreases significantly, especially when increasing the time lag value past 10. 

Therefore, we can see that Increasing the time lag reduces the error of the SHRED model as a result of the additional temporal context it provides. The time lag refers to the number of previous time steps or measurements used as input to predict the current time step. By including more past measurements as input, the model gains access to a broader historical context, enabling it to capture more complex temporal patterns and dependencies in the data. Furthermore, increasing the time lag can help in smoothing out the effects of noise or short-term fluctuations in the data as the model can average out noisy or erratic behavior, leading to more stable and robust predictions.

### 2. Do an analysis of the performance as a function of noise (add Gaussian noise to data)
The graph of the relative error associated with adding differing noise levels to the data is shown below:

![Screenshot 2023-05-22 at 11 25 36 PM](https://github.com/idane2309/EE399/assets/122940974/016bae51-55a0-4075-895f-061f143e9c38)


From the graph, we can see that between a noise level of 0.0 and 0.05, the relative error acts sporadically, spiking up to 0.0435 then back down to its lowest point at noise level = 0.05. However, past a noise level of 0.05, the relative error increases linearly with the noise level. 

Increasing the noise level initially can have a regularizing effect on the SHRED model, leading to a reduction in error. Regularization refers to a technique that helps prevent overfitting by introducing controlled amounts of noise or constraints to the model during training. The noise introduces some randomness, making the model more robust and less sensitive to individual data points or outliers. However, as the noise level increases beyond a certain threshold (in this case, 0.05), the noise becomes more disruptive and detrimental to the model's performance, increasing the distortion to the data, causing the error to rise linearly. 

### 3. Do an analysis of the performance as a function of the number of sensors
The graph of the relative error associated with differing the number of sensors is shown below:

![Screenshot 2023-05-22 at 11 25 54 PM](https://github.com/idane2309/EE399/assets/122940974/b767fa34-e28f-4b14-b14e-8009e3285815)


From the graph, we can see that the error drops significantly when going from 1 sensor to 2 sensors. Then after 2 sensors, the error still decreases linearly, however to a smaller extent than the initial drop from 1 sensor to 2 sensors. 

When increasing the number of sensors from 1 to 2, the error drop can be attributed to the improved information richness and diversity provided by having one additional sensor to contradict the first sensor. This could be the result of increased Spatial Coverage: With two sensors, the model gains access to measurements from different spatial locations within the system introducing additional perspectives and measurements to the underlying system

However, as the number of sensors continues to increase beyond a certain point, the error may start to decrease linearly. This can be attributed to the factors such as: Redundancy and Overlapping Information, Increased Noise and Complexity or Model Capacity and Overfitting.

## Sec. VI. Summary and Conclusion

In this project, I explored the application of SHRED (SHallow REcurrent Decoders) models for analyzing sea-surface temperature (SST) data. The objective was to understand the model's performance in relation to time lag, noise level, and the number of sensors.

I trained the SHRED model using SST data, randomly selecting three sensor locations and setting the trajectory length (lags) to 52. By generating input sequences, I evaluated the model's performance in terms of reconstruction and forecasting of sensor measurements. The model demonstrated promising results, capturing complex spatio-temporal patterns.

I investigated the impact of noise on the model's performance by introducing Gaussian noise at different levels. I found that initially, as the noise level increased, the error reduced. However, beyond a certain threshold (around 0.05), the error increased linearly. This suggests that the model's ability to handle noise diminishes after a certain point.

Furthermore, I explored the influence of the number of sensors on the model's performance. Increasing the number of sensors from 1 to 2 resulted in a significant drop in error, indicating the importance of information richness and spatial coverage. However, as the number of sensors increased further, the error reduction became linear, suggesting diminishing returns and potential overfitting.

Additionally, I examined the effect of different time lag values on the model's performance. Increasing the time lag led to a notable reduction in error, indicating that incorporating a longer history of sensor measurements improved the model's ability to capture temporal dependencies and make accurate predictions.

In conclusion, the SHRED model shows promise in analyzing SST data and making predictions based on sensor measurements. It is sensitive to factors such as time lag, noise level, and the number of sensors. Understanding these factors allows for informed decisions when applying the model to real-world scenarios.

