"""
STREESHIELD Phase 13D
AI Integration Helper

Connects the completed:
- AI Explanation Engine
- AI Report Generator
- Risk Assessment

with the existing Streamlit detection results.

IMPORTANT:
This module does NOT modify:
- CNN models
- 3D CNN models
- preprocessing
- dataset
- training
- Phase 11 modules
- Phase 12 modules
"""


# ==================================================
# IMPORT COMPLETED AI MODULES
# ==================================================

from explanation_engine import (
    AIExplanationEngine
)

from ai_report import (
    STREESHIELDReportGenerator
)

from risk_assesment import (
    STREESHIELDRiskAssessment
)


# ==================================================
# PHASE 13D INTEGRATION CLASS
# ==================================================

class STREESHIELDPhase13Integration:

    def __init__(self):

        self.project_name = "STREESHIELD"

        self.explanation_engine = (
            AIExplanationEngine()
        )

        self.report_generator = (
            STREESHIELDReportGenerator()
        )

        self.risk_assessment = (
            STREESHIELDRiskAssessment()
        )


    # ==================================================
    # GENERATE AI EXPLANATION
    # ==================================================

    def generate_ai_explanation(
        self,
        model_name,
        prediction,
        confidence,
        media_type
    ):
        """
        Generate an AI explanation from
        an existing model prediction.
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
        # PREDICTION EXPLANATION
        # ----------------------------------------------

        prediction_text = (
            self.explanation_engine
            .prediction_explanation(
                prediction,
                model_name
            )
        )


        # ----------------------------------------------
        # CONFIDENCE EXPLANATION
        # ----------------------------------------------

        confidence_text = (
            self.explanation_engine
            .confidence_interpretation(
                confidence
            )
        )


        # ----------------------------------------------
        # POSSIBLE INDICATORS
        # ----------------------------------------------

        indicators = (
            self.explanation_engine
            .possible_indicators(
                model_name,
                prediction
            )
        )


        return {

            "prediction_explanation":
                prediction_text,

            "confidence_interpretation":
                confidence_text,

            "possible_indicators":
                indicators,

            "media_type":
                media_type,

            "model":
                model_name,

            "prediction":
                prediction,

            "confidence":
                confidence
        }


    # ==================================================
    # GENERATE RISK ASSESSMENT
    # ==================================================

    def generate_risk_assessment(
        self,
        prediction,
        confidence
    ):
        """
        Calculate LOW / MEDIUM / HIGH risk.
        """

        risk = (
            self.risk_assessment
            .calculate_risk(
                prediction,
                confidence
            )
        )


        description = (
            self.risk_assessment
            .get_description(
                risk
            )
        )


        return {

            "risk": risk,

            "description":
                description
        }


    # ==================================================
    # GENERATE AI REPORT
    # ==================================================

    def generate_ai_report(
        self,
        model_name,
        prediction,
        confidence,
        media_type,
        ai_explanation
    ):
        """
        Generate the final AI report for
        the currently detected media.
        """

        report = (
            self.report_generator
            .generate_report(

                detection_type=media_type,

                model_name=model_name,

                prediction=prediction,

                confidence=confidence,

                ai_explanation=ai_explanation
            )
        )


        return report


    # ==================================================
    # COMPLETE AI ANALYSIS
    # ==================================================

    def analyze_detection(
        self,
        model_name,
        prediction,
        confidence,
        media_type
    ):
        """
        Perform complete Phase 13D AI processing.

        Returns:
        - AI explanation
        - risk assessment
        - AI report
        """

        prediction = str(
            prediction
        ).upper().strip()

        confidence = float(
            confidence
        )


        # ----------------------------------------------
        # STEP 1
        # AI EXPLANATION
        # ----------------------------------------------

        explanation_data = (
            self.generate_ai_explanation(

                model_name=model_name,

                prediction=prediction,

                confidence=confidence,

                media_type=media_type
            )
        )


        # ----------------------------------------------
        # CREATE REPORT EXPLANATION
        # ----------------------------------------------

        report_explanation = (

            explanation_data[
                "prediction_explanation"
            ]

            + " "

            + explanation_data[
                "confidence_interpretation"
            ]
        )


        # ----------------------------------------------
        # STEP 2
        # RISK
        # ----------------------------------------------

        risk_data = (
            self.generate_risk_assessment(

                prediction=prediction,

                confidence=confidence
            )
        )


        # ----------------------------------------------
        # STEP 3
        # AI REPORT
        # ----------------------------------------------

        report = (
            self.generate_ai_report(

                model_name=model_name,

                prediction=prediction,

                confidence=confidence,

                media_type=media_type,

                ai_explanation=report_explanation
            )
        )


        # ----------------------------------------------
        # RETURN COMPLETE RESULT
        # ----------------------------------------------

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

            "ai_explanation":
                explanation_data,

            "risk":
                risk_data,

            "report":
                report
        }


# ==================================================
# 13D TEST
# ==================================================

if __name__ == "__main__":

    print("\n========================================")

    print(
        "       STREESHIELD PHASE 13D"
    )

    print(
        "       AI INTEGRATION TEST"
    )

    print(
        "========================================"
    )


    # ----------------------------------------------
    # CREATE INTEGRATION
    # ----------------------------------------------

    integration = (
        STREESHIELDPhase13Integration()
    )


    # ----------------------------------------------
    # SAMPLE DETECTION RESULT
    # ----------------------------------------------

    result = (
        integration.analyze_detection(

            model_name="Basic CNN",

            prediction="REAL",

            confidence=68.17,

            media_type="Image"
        )
    )


    # ----------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------

    print("\n----------------------------------------")
    print("DETECTION")
    print("----------------------------------------")

    print(
        "Model      :",
        result["model"]
    )

    print(
        "Prediction :",
        result["prediction"]
    )

    print(
        f"Confidence : "
        f"{result['confidence']:.2f}%"
    )


    print("\n----------------------------------------")
    print("AI EXPLANATION")
    print("----------------------------------------")

    print(
        result[
            "ai_explanation"
        ][
            "prediction_explanation"
        ]
    )

    print()

    print(
        result[
            "ai_explanation"
        ][
            "confidence_interpretation"
        ]
    )


    print("\n----------------------------------------")
    print("RISK ASSESSMENT")
    print("----------------------------------------")

    print(
        "Risk       :",
        result[
            "risk"
        ][
            "risk"
        ]
    )

    print(
        "Description:",
        result[
            "risk"
        ][
            "description"
        ]
    )


    print("\n----------------------------------------")
    print("AI REPORT")
    print("----------------------------------------")

    print(
        result[
            "report"
        ]
    )


    print(
        "\n========================================"
    )

    print(
        "        13D INTEGRATION TEST COMPLETE"
    )

    print(
        "========================================"
    )