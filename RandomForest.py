
from BaseClassifier import BaseClassifier, logging
from sklearn.ensemble import RandomForestClassifier
from imports import np, shuffle

class LMIRandomForest(BaseClassifier):

    def __init__(self):
        pass

    def train(self, rf_model, X, y, descriptor_values):
        """ Trains a Random Forest model. Expects model_dict to contain hyperparameters *depth* and *n_est* (depth and number of estimators)
        
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
        assert "depth" in rf_model and "n_est" in rf_model
        logging.info(f'Training RF model with values of shape {X.shape}: max_depth={rf_model["depth"]} | n_est: {rf_model["n_est"]}')
        root = RandomForestClassifier(max_depth=rf_model["depth"], n_estimators=rf_model["n_est"])
        return super().train(root, X, y, descriptor_values)
    
    def add_zero_prob_classes(self, model, uniques, classes_votes, value="value_l"):
        zero_prob_classes = np.setdiff1d(model.classes_, uniques)
        shuffle(zero_prob_classes)
        for c in zero_prob_classes:
            classes_votes.append({value: int(c), "votes_perc": 0})
        return classes_votes
    
    def predict(self, query, model, encoder, value="value_l"):
        prob_distr = []
        query = query.reshape(1, -1)
        n_estimators = len(model.estimators_)
        
        for e in model.estimators_:
            prob_distr.append(int(e.predict(query)[0]))
        
        if encoder:
            prob_distr = encoder.inverse_transform(prob_distr)

        uniques, counts = np.unique(prob_distr, return_counts=True)
        classes_votes = []
        for u,c in zip(uniques, counts):
            classes_votes.append({value: int(u), "votes_perc": c / n_estimators})
        
        classes_votes = self.add_zero_prob_classes(model, uniques, classes_votes)
        classes_votes = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)

        return classes_votes
