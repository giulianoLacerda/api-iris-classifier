import logging
from v1.modules.classifier import Classifier


class IrisClassification:
    """This class implements an Iris classification routine.

    Attributes:
        logger (logging.Logger): Logger object.
        cfg (dict): A dictionary config.
        classifier (Classifier): Iris model classifier object.
    """

    def __init__(self, cfg, model):
        """
        Args:
            logger (logging.Logger): Logger object.
            cfg (dict): A dictionary config.
            classifier (Classifier): Iris model classifier object.
        """
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.classifier = Classifier(model)

    def main_routine(self, input):
        """Main routine for Iris classification.

        Args:
            input (dict): A dictionary with input data
                ({sepal_length: 1.0, sepal_width: 2.0, petal_length: 2.0, petal_width: 2.0}) the float
                values in cm unit.

        Returns:
            tuple(float, str): The confidence, new and predicted label.
        """

        if not input:
            self.logger.info("Input is empty")
            return None
        pred_conf, pred_label = self.classifier.predict(list(input.values()))
        return pred_conf, pred_label
