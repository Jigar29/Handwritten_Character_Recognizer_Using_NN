from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

#Cross-Validation Class
#Here K-folds cross validation is used 
class Cross_validation():
    folds = 0; 
    X = 0;
    y= 0; 
    skfold = 0; 
    model = 0;
    # Costructor 
    def __init__(self, no_of_folds, input_vector, label_vector, model):
        self.folds = no_of_folds;
        self.X = input_vector; 
        self.y = label_vector; 
        self.model = model; 
        self.skfold = StratifiedKFold(n_splits= self.folds, random_state= 43); 
        return; 
    # Method to predict the class of the data
    def predict(self):
        pred_y= 0; 
        desired_y=0;
        for train_index, test_index in self.skfold.split(self.X, self.y):
           Train_x = self.X[train_index]; 
           Test_x = self.X[test_index]; 
           Train_y = self.y[train_index];
           Test_y  = self.y[test_index];
           
           self.model.fit(Train_x, Train_y); 
           pred_y = np.append(pred_y, self.model.predict(Test_x));
           desired_y = np.append(desired_y, Test_y);
        return pred_y[1:], desired_y[1:]; 
    # Method to print the Permonace of the Model 
    def score(self):
        pred_y, desired_y = self.predict(); 
        
        for i in range(0, self.folds):
            start_pos = i*(len(pred_y)//self.folds); 
            end_pos = (i+1)*(len(pred_y)//self.folds);
            ncorrect = sum(pred_y[start_pos:end_pos] == desired_y[start_pos:end_pos]);
            print('Accuracy is = '+ str(ncorrect/ len(pred_y[start_pos:end_pos])))
        return 
    
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

#Training the Model with the SGD Classifier 
SGD_model = SGDClassifier(random_state=32, max_iter= 100, tol = 0); 
#SGD_model.fit(Train_X, Train_y); 

# Cross Validation 
#No of K = 3; 
cross_val = Cross_validation(3, Train_X, Train_y, model= SGD_model);
cross_val.score()

#Confusion Matrix 
x, y = cross_val.predict();
cof_mat = confusion_matrix(y, x)