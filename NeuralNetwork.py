
from BaseClassifier import BaseClassifier, logging
from utils import np, label_encode_data, get_class_weights_dict

class LMINeuralNetwork(BaseClassifier):

    def __init__(self):
        pass

    def train(self, nn_model, X, y, descriptor_values, input_data_shape=None, output_data_shape=None, loss='sparse_categorical_crossentropy', metric='accuracy'):
        """ Trains a Neural Network. Expects model_dict to contain hyperparameters *opt* and *ep* (depth and number of estimators)
        
        Parameters
        -------
        rf_model: Dict
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
        assert "opt" in nn_model and "model" in nn_model and "ep" in nn_model
        logging.info(f'Training NN with model: {nn_model["model"]}, optimizer: {str(nn_model["opt"])} and epochs: {nn_model["ep"]}')
        is_multi = True if len(y.shape) > 1 else False
        
        if is_multi:
            y_weights_input = np.unique(np.argmax(y, axis=1))
        else:
            y_weights_input = y

        d_class_weights = get_class_weights_dict(y_weights_input)
        y, encoder = label_encode_data(y)
        
        if input_data_shape and output_data_shape:
            model = nn_model["model"](input_data_shape=input_data_shape, output_data_shape=output_data_shape)
        else:
            model = nn_model["model"]()
        model.compile(loss=loss, metrics=[metric], optimizer=nn_model["opt"])
        model.fit(X, y, epochs=nn_model["ep"], class_weight=d_class_weights)
        
        predictions = model.predict(X)
        predictions = [np.argmax(p) for p in predictions]
        predictions = encoder.inverse_transform(predictions)
        return model, predictions, encoder 

    def predict(self, query, model, encoder):
        prob_distr = model.predict(query)[0]
        return super().predict(prob_distr, encoder)