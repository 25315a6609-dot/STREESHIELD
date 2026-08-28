import tensorflow as tf


# ==================================================
# 3D CNN CONNECTOR
# ==================================================

class CNN3DConnector:

    def __init__(
        self,
        model_path
    ):

        self.model = tf.keras.models.load_model(
            model_path
        )


    # ==================================================
    # CONVERT MODEL OUTPUT
    # ==================================================

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
            "model": "3D CNN",
            "prediction": prediction,
            "confidence": confidence,
            "raw_probability": probability
        }