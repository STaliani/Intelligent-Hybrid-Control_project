import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class GPR:
    def __init__(self, kernel):
        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        return
    
    def train_regressor(self, dataset):
        m, _ = dataset.shape
        X = dataset[2:,:]
        y = dataset[:2,:] # ground truth
        print(y.shape)
        self.model.fit(X, y)
        print("Training completed")
        #X_test = dataset[list(range(3,m,2)),:]
        #y_test = dataset[1,:]
        #print("Score: ", self.model.score(X_test[0,:], y_test))
        print("Score: ", self.model.score(X, y))
        
        


    def extract_evolution_from_measure(self, noisy_sequence: np.ndarray):
        prediction = self.model.predict(noisy_sequence.transpose())
        print("Prediction:\n", prediction)
    