"""
STREESHIELD Phase 11E
AI Explanation Engine

This module explains existing CNN/3D CNN predictions.
It does not retrain or modify the models.
"""


class AIExplanationEngine:

    def __init__(self):

        self.project_name = "STREESHIELD"


    # ==================================================
    # CONFIDENCE INTERPRETATION
    # ==================================================

    def confidence_interpretation(
        self,
        confidence
    ):

        confidence = float(
            confidence
        )

        if confidence >= 80:

            return (
                "The model is highly confident in "
                "this prediction."
            )

        elif confidence >= 60:

            return (
                "The model has moderate confidence "
                "in this prediction."
            )

        else:

            return (
                "The model has low confidence. "
                "This result should be interpreted cautiously."
            )


    # ==================================================
    # PREDICTION EXPLANATION
    # ==================================================

    def prediction_explanation(
        self,
        prediction,
        model_name
    ):

        prediction = str(
            prediction
        ).upper()

        if prediction == "FAKE":

            if model_name == "Basic CNN":

                return (
                    "The Basic CNN classified the image "
                    "as FAKE because its learned image-level "
                    "features were more consistent with the "
                    "FAKE class."
                )

            elif model_name == "3D CNN":

                return (
                    "The 3D CNN classified the video as FAKE "
                    "because its learned spatial-temporal "
                    "features were more consistent with the "
                    "FAKE class."
                )

            return (
                "The model identified patterns associated "
                "with the FAKE class."
            )

        else:

            if model_name == "Basic CNN":

                return (
                    "The Basic CNN classified the image "
                    "as REAL because its learned image-level "
                    "features were more consistent with the "
                    "REAL class."
                )

            elif model_name == "3D CNN":

                return (
                    "The 3D CNN classified the video as REAL "
                    "because its learned spatial-temporal "
                    "features were more consistent with the "
                    "REAL class."
                )

            return (
                "The model identified patterns associated "
                "with the REAL class."
            )


    # ==================================================
    # POSSIBLE INDICATORS
    # ==================================================

    def possible_indicators(
        self,
        model_name,
        prediction
    ):

        if model_name == "Basic CNN":

            indicators = [
                "Facial texture patterns",
                "Image-level facial inconsistencies",
                "Blending or boundary artifacts",
                "Compression-related anomalies"
            ]

        elif model_name == "3D CNN":

            indicators = [
                "Frame-to-frame appearance changes",
                "Temporal inconsistencies",
                "Spatial-temporal facial patterns",
                "Possible motion or expression inconsistencies"
            ]

        else:

            indicators = [
                "Model-specific learned features"
            ]

        if str(prediction).upper() == "FAKE":

            indicators.append(
                "Patterns associated with the FAKE class "
                "were detected by the model."
            )

        else:

            indicators.append(
                "No strong FAKE-class pattern was detected "
                "by the selected model."
            )

        return indicators


    # ==================================================
    # GENERATE COMPLETE EXPLANATION
    # ==================================================

    def generate_explanation(
        self,
        analysis_input
    ):

        data = analysis_input.to_dict()

        model_name = data[
            "model_name"
        ]

        prediction = data[
            "prediction"
        ]

        confidence = data[
            "confidence"
        ]

        media_type = data[
            "media_type"
        ]

        explanation = (
            self.prediction_explanation(
                prediction,
                model_name
            )
        )

        confidence_text = (
            self.confidence_interpretation(
                confidence
            )
        )

        indicators = (
            self.possible_indicators(
                model_name,
                prediction
            )
        )

        return {

            "project":
                self.project_name,

            "model":
                model_name,

            "media_type":
                media_type,

            "prediction":
                prediction,

            "confidence":
                confidence,

            "confidence_interpretation":
                confidence_text,

            "prediction_explanation":
                explanation,

            "possible_indicators":
                indicators,

            "warning":
                (
                    "These indicators describe general "
                    "patterns the model may rely on. "
                    "They do not prove that a specific "
                    "artifact is present in the uploaded media."
                )
        }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    from analysis_input import AIAnalysisInput


    analysis_input = AIAnalysisInput(

        model_name="Basic CNN",

        prediction="REAL",

        confidence=68.17,

        media_type="Image",

        raw_probability=0.3182801306,

        media_info={
            "image_size": "128x128",
            "face_detected": True
        }
    )


    engine = AIExplanationEngine()


    result = engine.generate_explanation(
        analysis_input
    )


    print("\n========================================")
    print("       PHASE 11E EXPLANATION TEST")
    print("========================================")


    for key, value in result.items():

        print(
            f"\n{key}:"
        )

        print(
            value
        )


    print("\n========================================")
    print("          11E TEST COMPLETE")
    print("========================================")