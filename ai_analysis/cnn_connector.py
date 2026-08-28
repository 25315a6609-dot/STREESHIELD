import tensorflow as tf


# ==================================================
# BASIC CNN CONNECTOR
# ==================================================

class BasicCNNConnector:

    def __init__(
        self,
        model_path
    ):

        self.model = tf.keras.models.load_model(
            model_path
        )


    def analyze_output(
        self,
        probability
    ):

        probability = float(
            probability
        )

        if probability >= 0.5:

            prediction = "FAKE"

            confidence = (
                probability * 100
            )

        else:

            prediction = "REAL"

            confidence = (
                (1 - probability) * 100
            )

        return {
            "model": "Basic CNN",
            "prediction": prediction,
            "confidence": confidence,
            "raw_probability": probability
        }