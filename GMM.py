
from BaseClassifier import BaseClassifier, logging
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel(BaseClassifier):

    def __init__(self):
        pass

    def train(self, gmm_model, X, y, descriptor_values):
        """ Trains a Gaussian mixture model. Expects model_dict to contain hyperparameters *comp* and *cov_type* (n. of components and covariance type)
        
        Parameters
        -------
        gmm_model: Dict
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
        assert "comp" in gmm_model and "cov_type" in gmm_model
        logging.info(f'Training GMM model with values of shape {X.shape}: n. of clusters={gmm_model["comp"]} | covariance type={gmm_model["cov_type"]}')
        if X.shape[0] <= gmm_model["comp"]:
            previous_gmm_comp = gmm_model["comp"]
            gmm_model["comp"] = X.shape[0] // 2
            logging.warn(f"Reducing the number of components from {previous_gmm_comp} to {gmm_model['comp']} since the number of\
                           training samples ({X.shape[0]}) is less than {previous_gmm_comp}")
        root = GaussianMixture(n_components=gmm_model["comp"], covariance_type=gmm_model["cov_type"])
        return super().train(root, X, y=None, descriptor_values=descriptor_values)
    
    def predict(self, query, model, encoder, value="value_l"):
        prob_distr = model.predict_proba(query)[0]
        return super().predict(prob_distr, encoder)
