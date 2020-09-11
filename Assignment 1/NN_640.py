import numpy as np
import matplotlib.pyplot as plt



# HYPERPARAMETERS

# Datasets
LINEAR = ["DataFor640/dataset1/LinearX.csv", "DataFor640/dataset1/LinearY.csv"]
NON_LINEAR = ["DataFor640/dataset1/NonlinearX.csv", "DataFor640/dataset1/NonlinearY.csv"]
DIGIT_TRAIN = ["DataFor640/dataset2/Digit_X_train.csv", "DataFor640/dataset2/Digit_y_train.csv"]
DIGIT_TEST = ["DataFor640/dataset2/Digit_X_test.csv", "DataFor640/dataset2/Digit_y_test.csv"]

DATASET = LINEAR    # Choose one of the above
N_FOLDS = 5     # Number of folds for the cross validation

# Single layer fully connected feed forward neural network
N_NEURONS = 10   # Number of neurons in the hidden layer
LAMBDA = 0.5     # Value of lambda for the l2 regularization
LEARNING_RATE = 0.0025     # Learning rate for the gradient descent
N_EPOCHS = 2000   # Number of epoch
ACTIVATION_FUNCTION = "sigmoid"     # Choose between "sigmoid", "tanh", "arctan" and "relu"



class ActivationFunctions:
    def sigmoid(self, X):
        """
            Implementation of the Sigmoid function
            Can be used as an activation function
        """
        return (1 / (1 + np.exp(-X)))


    def d_sigmoid(self, X):
        """
            The derivative of the Sigmoid function
        """
        s = ActivationFunctions().sigmoid(X)
        return s * (1 - s)
    

    def tanh(self, X):
        """
            Implementation of the Hyperbolic Tangent function
            Can be used as an activation function
        """
        return np.tanh(X)


    def d_tanh(self, X):
        """
            The derivative of the Hyperbolic Tangent function
        """
        return (1 - np.tanh(X)**2)

    
    def arctan(self, X):
        """
            Implementation of the Arcus Tangent function (inverse of the Tangent)
            Can be used as an activation function
        """
        return np.arctan(X)


    def d_arctan(self, X):
        """
            The derivative of the Arcus Tangent function (inverse of the Tangent)
        """
        return (1 + X**2)**-1

    
    def relu(self, X):
        """
            Implementation of the ReLU function
            Can be used as an activation function
        """
        return np.maximum(0, X)


    def d_relu(self, X):
        """
            The derivative of the ReLU function
        """
        return 1 if X > 0 else 0


class NeuralNetwork:
    def __init__(self, NNodes, activate, deltaActivate):
        self.NNodes = NNodes # the number of nodes in the hidden layer
        self.activate = activate # a function used to activate
        self.deltaActivate = deltaActivate # the derivative of activate
    

    def fit(self, X, Y, learningRate, epochs, regLambda):
        """
        This function is used to train the model.
        Parameters
        ----------
        X : numpy matrix
            The matrix containing sample features for training.
        Y : numpy array
            The array containing sample labels for training.
        Returns
        -------
        None
        """   
        attributes = X.shape[1]

        self.wh = np.random.rand(attributes, self.NNodes) * 0.01
        self.bh = np.random.randn(self.NNodes)
        self.wo = np.random.rand(self.NNodes, Y.shape[1]) * 0.01
        self.bo = np.random.randn(Y.shape[1])
           
        for epoch in range(epochs):
            ah, ao, zh = self.forward(X)
            dcost_wh, dcost_bh, dcost_wo, dcost_bo = self.backpropagate(X,Y,ah,ao,zh)

            self.wh -= (learningRate * dcost_wh) + ((regLambda*self.wh)/X.shape[0]) 
            self.bh -= learningRate * dcost_bh.sum(axis=0) 
            self.wo -= (learningRate * dcost_wo) + ((regLambda*self.wo)/X.shape[0]) 
            self.bo -= learningRate * dcost_bo.sum(axis=0)
            
            if epoch % 200 == 0:
                loss = self.getCost(Y,ao,regLambda)
                print('Loss function value: ', loss)


    def predict(self, X):
        """
        Predicts the labels for each sample in X.
        Parameters
        X : numpy matrix
            The matrix containing sample features for testing.
        Returns
        -------
        YPredict : numpy array
            The predictions of X.
        ----------
        """
        ah, ao, zh = self.forward(X)
        b = np.zeros_like(ao)
        b[np.arange(len(ao)), ao.argmax(1)] = 1

        return b


    def forward(self, X):
        zh = np.dot(X, self.wh) + self.bh
        ah = self.activate(zh)
        zo = np.dot(ah, self.wo) + self.bo
        ao = softmax(zo)
        
        return ah, ao, zh

        
    def backpropagate(self, X, Y, ah, ao, zh):
        # Loss wrt Wo
        dcost_dzo = ao - Y
        dzo_dwo = ah
        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)
        
        #Loss wrt bo
        dcost_bo = dcost_dzo
        
        #Loss wrt wh
        dzo_dah = self.wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)        
        dah_dzh = self.deltaActivate(zh)
        dzh_dwh = X
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        
        #Loss wrt bh
        dcost_bh = dcost_dah * dah_dzh
        
        return dcost_wh, dcost_bh, dcost_wo, dcost_bo
        

    def getCost(self, YTrue, YPredict, Lambda):
        # Compute loss / cost in terms of crossentropy.
        loss = np.sum(-YTrue * np.log(YPredict))
        reg_parameter = (np.sum(self.wh**2)+np.sum(self.wo**2))*(Lambda/X.shape[0])
        loss = loss + reg_parameter
        
        return loss


def getData(dataDir, is_one_hot=True):
    '''
    Returns
    -------
    X : numpy matrix
        Input data samples.
    Y : numpy array
        Input data labels.
    '''
    """
    X = pd.read_csv(features_file_path, header = None)
    X = np.array(X)
    y = pd.read_csv(labels_file_path, header = None)
    Y = np.array(y)
    """
    X = np.loadtxt(dataDir[0], delimiter=",")
    Y = np.loadtxt(dataDir[1], delimiter=",")

    #Conversion to one-hot vectors
    if is_one_hot:
        classes_one_hot = np.zeros((Y.shape[0], int(np.max(Y)) + 1))
        for i, class_index in enumerate(Y):
            classes_one_hot[i, int(class_index)] = 1

    return X, classes_one_hot


def splitData(X, Y, k=5):
    '''
    Returns
    -------
    result : List[[train, test]]
        "train" is a list of indices corresponding to the training samples in the data.
        "test" is a list of indices corresponding to the testing samples in the data.
        For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
        sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
        are for training.
    ''' 
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    folds = np.array_split(data, k)
    train = np.asarray(folds[:-1]).reshape((k-1)*int(len(data)/k), data.shape[1])
    test = np.asarray(folds[-1])

    return train, test


def plotDecisionBoundary(model, X, Y):
   
       
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z[:,1:]   
    Z = Z.reshape(x1_array.shape)
    Y_drop = Y[:,1:]
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    zeros = X[(np.where(Y_drop==0))[0]]
    plt.scatter(zeros[:, 0], zeros[:, 1] ,c = ["yellow"])
    ones = X[(np.where(Y_drop==1))[0]]
    plt.scatter(ones[:, 0], ones[:, 1] ,c = ["black"])
    plt.show()


def softmax(X):
    """
        Implementation of the softmax function
        It is applied to the output layer of the neural network
    """
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)


def train(XTrain, YTrain, **kwargs):
    """
    This function is used for the training phase.
    Parameters
    ----------
    XTrain : numpy matrix
        The matrix containing samples features (not indices) for training.
    YTrain : numpy array
        The array containing labels for training.
    args : List
        The list of parameters to set up the NN model.
    Returns
    -------
    NN : NeuralNetwork object
        This should be the trained NN object.
    """
    NN = NeuralNetwork(kwargs["NNodes"], kwargs["activate"], kwargs["deltaActivate"])
    NN.fit(XTrain, YTrain, kwargs["learningRate"], kwargs["epochs"], kwargs["regLambda"])
    
    
    return NN


def test(XTest, model):
    """
    This function is used for the testing phase.
    Parameters
    ----------
    XTest : numpy matrix
        The matrix containing samples features (not indices) for testing.
    model : NeuralNetwork object
        This should be a trained NN model.
    Returns
    -------
    YPredict : numpy array
        The predictions of X.
    """
    return model.predict(XTest)


def getConfusionMatrix(YTrue, YPredict):
    confusion = np.zeros((YTrue.shape[1],YTrue.shape[1]))
    
    for i in range(YTrue.shape[0]):
        m = np.where(YTrue[i,:] == 1)
        n = np.where(YPredict[i,:] == 1)
        confusion[n,m] =confusion[n,m] + 1
    
    print("\n\n\n")
    print("------------CONFUSION MATRIX----------")
    print(confusion)
    return confusion    


def getPerformanceScores(YTrue, YPredict):
    confusion_matrix = getConfusionMatrix(YTrue, YPredict)
    accuracy, precision, recall, f_1 = [], [], [], []
    
    for i in range(YTrue.shape[1]):
        true_positive, true_negative, false_positive, false_negative = 0, 0 , 0, 0

        for j in range(YTrue.shape[0]):
            if YPredict[j, i] == 1 and YTrue[j, i] == 1:
                true_positive += 1
            elif (YPredict[j, i] == 1 and YTrue[j, i] == 0):
                false_positive += 1
            elif (YPredict[j, i] == 0 and YTrue[j, i] == 1):
                false_negative += 1
            elif (YPredict[j, i] == 0 and YTrue[j, i] == 0):
                true_negative += 1

        accuracy.append((true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative))
        precision.append(true_positive / (true_positive + false_positive))
        recall.append(true_positive / (true_positive + false_negative))
        f_1.append(2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1]))

    return {
            "cm" : confusion_matrix, 
            "accuracy" : np.mean(accuracy), 
            "precision" : np.mean(precision), 
            "recall" : np.mean(recall), 
            "f1" : np.mean(f_1)
        }



np.random.seed(42)

a_f = ActivationFunctions()

if ACTIVATION_FUNCTION == "sigmoid":
    ACTIVATION_FUNCTION = a_f.sigmoid
    D_ACTIVATION_FUNCTION = a_f.d_sigmoid
elif ACTIVATION_FUNCTION == "tanh":
    ACTIVATION_FUNCTION = a_f.tanh
    D_ACTIVATION_FUNCTION = a_f.d_tanh
elif ACTIVATION_FUNCTION == "arctan":
    ACTIVATION_FUNCTION = a_f.arctan
    D_ACTIVATION_FUNCTION = a_f.d_arctan
elif ACTIVATION_FUNCTION == "relu":
    ACTIVATION_FUNCTION = a_f.relu
    D_ACTIVATION_FUNCTION = a_f.d_relu
else:
    raise ValueError("Unknown activation function")





plot = True
X, Y = getData(DATASET)
if DATASET == DIGIT_TRAIN:
    plot = False
    X_test, Y_test = getData(DIGIT_TEST)
elif DATASET == DIGIT_TEST:
    raise ValueError("This is a dataset for training. Use DIGIT_TRAIN instead")
else:
    train_set, test_set = splitData(X, Y, N_FOLDS)    
    X, Y = train_set[:, :X.shape[1]], train_set[:, X.shape[1]:]
    X_test, Y_test = test_set[:, :X.shape[1]], test_set[:, X.shape[1]:]



model = train(X, Y, **{"NNodes" : N_NEURONS, "activate" : ACTIVATION_FUNCTION, "deltaActivate" : D_ACTIVATION_FUNCTION, "learningRate" : LEARNING_RATE, "epochs" : N_EPOCHS, "regLambda" : LAMBDA, "plot" : plot})
predictions = test(X, model)
metrics_training = getPerformanceScores(Y, predictions)

predictions = test(X_test, model)
metrics_testing = getPerformanceScores(Y_test, predictions)

if plot==True:
        plotDecisionBoundary(model, X_test, Y_test)

cm = getConfusionMatrix(Y_test, predictions)
print(cm)


print("-------------TRAINING METRICS ------------------")
print("Accuracy :" + str(metrics_training["accuracy"]*100) + " \nPrecision : " + str(metrics_training["precision"]) + " \nRecall : " + str(metrics_training["recall"]) + " \nF1 : " + str(metrics_training["f1"])+ "\n\n\n\n\n")

print("-------------TESTING METRICS ------------------")
print("Accuracy :" + str(metrics_testing["accuracy"]*100) + " \nPrecision : " + str(metrics_testing["precision"]) + " \nRecall : " + str(metrics_testing["recall"]) + " \nF1 : " + str(metrics_testing["f1"]))


