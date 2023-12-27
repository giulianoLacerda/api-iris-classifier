import pickle


class Classifier:
    """This class implements an classifier.

    Attributes:
        model (mlflow.pyfunc.PyFuncModel): Model loaded from mlflow.
    """

    def __init__(self, model):
        if isinstance(model, str):
            self.load(model)
        else:
            self.model = model

    def load(self, input_path):
        """Load model from input path.

        Args:
            input_path (str): Path to serialized .pkl file.
        """
        with open(input_path, "rb") as file:
            self.model = pickle.load(file)

    def _get_class(self, conf_list, pred_conf):
        """Mapping the pred index class to label.

        Args:
            conf_list (list): List with confidences for each label.

        Returns:
            str: Returns a string label.
        """
        return self.model._model_impl.sklearn_model.classes_[list(conf_list).index(pred_conf)]

    def _predict_proba(self, input):
        """Predict the list of probabilitys for a input.

        Args:
            input (list[float]): Input iris features (sepal_length, sepal_width, petal_length, petal_width)

        Returns:
            list: Returns the predict probability for each label.
        """
        return self.model._model_impl.sklearn_model.predict_proba([input])[0]

    def _pre_predict(self, input):
        """
        The _pre_predict function takes in an input and returns a prediction confidence and label.

        Args:
            input (list): Pass the data to be predicted

        Returns:
            tuple(float, str): The confidence, new and predicted label.
        """
        confidences_list = self._predict_proba(input)
        pred_conf = confidences_list.max()
        pred_label = self._get_class(confidences_list, pred_conf)
        return pred_conf, pred_label

    def predict(self, input):
        """Predict class and confidence for a input iris.

        Args:
            input (list[float]): Input iris features (sepal_length, sepal_width, petal_length, petal_width)

        Returns:
            tuple(float, str): The confidence, new and predicted label.
        """
        if not input:
            return None
        pred_conf, pred_label = self._pre_predict(input)
        return pred_conf, pred_label
