from sklearn.datasets.mldata import fetch_mldata
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

dataset = fetch_mldata('MNIST original', data_home= 'A:/Dropbox/Jigar/SJSU_Data/EE258/Datasets/Mnist')

#Importing the Data from the dataset 
X = dataset["data"];
y = dataset["target"]; 

#Printing random image at 10062 position to verify the data fetch was successful or not 
element_num  = 10062
temp = X[element_num].reshape(28,28)
plt.imshow(temp)
plt.show()
print(y[element_num])              #Printing the label associated with the prinnted image 

#Separating the trainig and test data
Train_X = X[:60000]; 
Train_y = y[:60000];
Test_X  = X[60000:]; 
Test_y  = y[60000:]; 

#Training data and the test data are separetd properly, 
#We dont need to worry about them. But as we are going to perform a cross-validation on the tranning set 
#lets make shufful it so that the data gets equally distributed and none of the data digits escape any of the 
#validation set or the traning set, as some algorithm only performs well on the equally distributed data. 
reviced_index = np.random.permutation(Train_X.shape[0])
Train_X, Train_y = Train_X[reviced_index], Train_y[reviced_index]

#Training the Model with the MLP Classifier with one hidden layer fully connected 
ML_model = MLPClassifier(alpha=1e-5, learning_rate_init=0.001, max_iter= 800, hidden_layer_sizes=(100, 100), activation='logistic', early_stopping= True, random_state=1);
ML_model.n_layers_ = 3;
ML_model.n_outputs_ = 10;

ML_model.fit(Train_X, Train_y);                    

# Cross Validation for predicting training error
#No of K = 10; 
score = cross_val_score(ML_model, Train_X, Train_y, cv= 3)
training_accuracy = score.mean() * 100;
print('Traning Accuracy is = ' + str(training_accuracy) + '%')

# Test Error calculation 
Test_pred_y = ML_model.predict(Test_X);
test_accuracy = (sum(Test_pred_y == Test_y)/ len(Test_pred_y))*100; 
print('Generalization Accuracy = ' + str(test_accuracy)+ '%');

#computing loss
print('Cost = ' + str(ML_model.loss_));
    
#Confusion Matrix 
conf_mat = confusion_matrix(Test_y, Test_pred_y)
print('****************Confusion Matrix*******************')
print(conf_mat)