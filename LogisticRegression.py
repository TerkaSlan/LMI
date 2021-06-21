from imports import np
from sklearn.linear_model import LogisticRegression
from BaseClassifier import BaseClassifier, logging
from utils import get_class_weights_dict

def custom_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

class BarebonesLogisticRegression(LogisticRegression):

    def predict_proba_single(self, x):
        return custom_softmax(np.dot(self.coef_, x[0]) + self.intercept_)

class LMILogisticRegression(BaseClassifier):
    
    def __init__(self):
        pass
    
    def train(self, lr_model, X, y, descriptor_values):
        """ Trains a Logistic regression model. Expects model_dict to contain hyperparameter *ep* (number of epochs)
        
        Parameters
        -------
        lr_model: Dict
            Dictionary of model specification
        X: Numpy array
            Training values
        y: Numpy array
            Training labels 
        
        Returns
        -------
        predictions: Numpy array
            Array of model predictions
            
        encoder: LabelEncoder
            Mapping of actual labels to labels used in training
        """

        assert "ep" in lr_model
        logging.info(f'Training LogReg model with values of shape {X.shape}: epochs={lr_model["ep"]}')
        
        d_class_weights = get_class_weights_dict(y)
        model = BarebonesLogisticRegression(max_iter=lr_model["ep"], class_weight=d_class_weights)
        return super().train(model, X, y, descriptor_values)
    
    def predict(self, query, model, encoder):
        prob_distr = model.predict_proba_single(query)
        return super().predict(prob_distr, encoder)