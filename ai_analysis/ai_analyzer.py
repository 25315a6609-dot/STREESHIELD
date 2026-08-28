"""
STREESHIELD AI Analysis Module

Phase 11A:
Separate AI explanation layer.

This module does NOT modify or retrain the existing
CNN or 3D CNN models.

It receives model results and produces structured
analysis that can later be connected to an external
AI model/API in Phase 11E.
"""


# ==================================================
# AI ANALYSIS CLASS
# ==================================================

class STREESHIELDAnalyzer:
    """
    AI analysis layer for STREESHIELD.

    Existing ML models remain completely separate.
    """

    def __init__(self):
        self.project_name = "STREESHIELD"

    # ==================================================
    # BASIC RESULT ANALYSIS
    # ==================================================

    def analyze_prediction(
        self,
        model_name,
        prediction,
        confidence,
        media_type
    ):
        """
        Analyze an existing model prediction.

        Parameters
        ----------
        model_name : str
            "Basic CNN" or "3D CNN"

        prediction : str
            "REAL" or "FAKE"

        confidence : float
            Confidence percentage, 0-100

        media_type : str
            "Image" or "Video"
        """

        prediction = str(
            prediction
        ).upper().strip()

        model_name = str(
            model_name
        ).strip()

        media_type = str(
            media_type
        ).strip()

        confidence = float(
            confidence
        )

        # ----------------------------------------------
        # VALIDATION
        # ----------------------------------------------

        if prediction not in [
            "REAL",
            "FAKE"
        ]:

            raise ValueError(
                "Prediction must be REAL or FAKE."
            )

        if not 0 <= confidence <= 100:

            raise ValueError(
                "Confidence must be between 0 and 100."
            )

        # ----------------------------------------------
        # INTERPRET CONFIDENCE
        # ----------------------------------------------

        confidence_level = (
            self.interpret_confidence(
                confidence
            )
        )

        # ----------------------------------------------
        # BUILD EXPLANATION
        # ----------------------------------------------

        if prediction == "FAKE":

            explanation = (
                f"The {model_name} classified the "
                f"uploaded {media_type.lower()} as FAKE "
                f"with {confidence:.2f}% confidence."
            )

        else:

            explanation = (
                f"The {model_name} classified the "
                f"uploaded {media_type.lower()} as REAL "
                f"with {confidence:.2f}% confidence."
            )

        # ----------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------

        if prediction == "FAKE":

            interpretation = (
                "The model detected patterns that are "
                "more consistent with the FAKE class "
                "than the REAL class."
            )

        else:

            interpretation = (
                "The model detected patterns that are "
                "more consistent with the REAL class "
                "than the FAKE class."
            )

        # ----------------------------------------------
        # POSSIBLE INDICATORS
        # ----------------------------------------------

        indicators = self.get_possible_indicators(
            model_name,
            prediction
        )

        return {
            "project": self.project_name,
            "model": model_name,
            "media_type": media_type,
            "prediction": prediction,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "explanation": explanation,
            "interpretation": interpretation,
            "possible_indicators": indicators
        }

    # ==================================================
    # CONFIDENCE INTERPRETATION
    # ==================================================

    def interpret_confidence(
        self,
        confidence
    ):
        """
        Convert numerical confidence into
        a simple interpretation.

        This is an interpretation layer, not
        a calibrated probability guarantee.
        """

        confidence = float(
            confidence
        )

        if confidence >= 80:

            return "High"

        elif confidence >= 60:

            return "Moderate"

        else:

            return "Low"

    # ==================================================
    # POSSIBLE INDICATORS
    # ==================================================

    def get_possible_indicators(
        self,
        model_name,
        prediction
    ):
        """
        Return possible visual/temporal indicators.

        These are potential indicators, not proof
        that a specific artifact exists in the input.
        """

        indicators = []

        if model_name == "Basic CNN":

            indicators.extend([
                "Facial texture inconsistencies",
                "Unusual image-level facial patterns",
                "Blending or boundary artifacts",
                "Compression-related visual anomalies"
            ])

        elif model_name == "3D CNN":

            indicators.extend([
                "Temporal inconsistencies across frames",
                "Frame-to-frame facial changes",
                "Spatial-temporal pattern differences",
                "Inconsistent facial motion or appearance"
            ])

        else:

            indicators.append(
                "Model-specific visual features"
            )

        if prediction == "REAL":

            indicators.append(
                "No strong FAKE-class pattern was detected "
                "by the selected model."
            )

        else:

            indicators.append(
                "The model found patterns associated "
                "with the FAKE class."
            )

        return indicators


# ==================================================
# QUICK TEST
# ==================================================

if __name__ == "__main__":

    analyzer = STREESHIELDAnalyzer()

    result = analyzer.analyze_prediction(
        model_name="Basic CNN",
        prediction="FAKE",
        confidence=73.50,
        media_type="Image"
    )

    print("\n========================================")
    print("      STREESHIELD AI ANALYSIS TEST")
    print("========================================")

    for key, value in result.items():

        print(
            f"\n{key}:"
        )

        print(
            value
        )

    print("\n========================================")
    print("           11A TEST COMPLETE")
    print("========================================")