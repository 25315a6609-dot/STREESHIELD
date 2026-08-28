"""
STREESHIELD Phase 13B
Combined AI Report

Combines:
- Basic CNN prediction
- Basic CNN confidence
- 3D CNN prediction
- 3D CNN confidence
- AI explanation

This module does NOT modify the existing
CNN, 3D CNN, preprocessing, or AI modules.
"""


# ==================================================
# REPORT GENERATOR
# ==================================================

class STREESHIELDCombinedReport:

    def __init__(self):

        self.project_name = "STREESHIELD"


    # ==================================================
    # GENERATE COMBINED REPORT
    # ==================================================

    def generate_report(
        self,
        cnn_prediction,
        cnn_confidence,
        cnn3d_prediction,
        cnn3d_confidence,
        ai_explanation
    ):

        # ----------------------------------------------
        # CONVERT VALUES
        # ----------------------------------------------

        cnn_prediction = str(
            cnn_prediction
        ).upper()

        cnn3d_prediction = str(
            cnn3d_prediction
        ).upper()

        cnn_confidence = float(
            cnn_confidence
        )

        cnn3d_confidence = float(
            cnn3d_confidence
        )

        ai_explanation = str(
            ai_explanation
        )


        # ----------------------------------------------
        # BUILD REPORT
        # ----------------------------------------------

        report = f"""
========================================
        STREESHIELD AI REPORT
========================================

Project        : STREESHIELD

----------------------------------------
BASIC CNN RESULT
----------------------------------------

Prediction     : {cnn_prediction}
Confidence     : {cnn_confidence:.2f}%


----------------------------------------
3D CNN RESULT
----------------------------------------

Prediction     : {cnn3d_prediction}
Confidence     : {cnn3d_confidence:.2f}%


----------------------------------------
AI EXPLANATION
----------------------------------------

{ai_explanation}


----------------------------------------
IMPORTANT NOTE
----------------------------------------

The predictions are based on patterns
learned by the detection models.

Model confidence represents the strength
of the model output and is not a guarantee
of correctness.

A FAKE prediction should not be interpreted
as definitive proof of manipulation without
additional evidence.


========================================
        END OF AI REPORT
========================================
"""

        return report


# ==================================================
# 13B TEST
# ==================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       STREESHIELD PHASE 13B")
    print("       COMBINED AI REPORT TEST")
    print("========================================")


    # ----------------------------------------------
    # SAMPLE CNN RESULT
    # ----------------------------------------------

    cnn_prediction = "FAKE"

    cnn_confidence = 87.42


    # ----------------------------------------------
    # SAMPLE 3D CNN RESULT
    # ----------------------------------------------

    cnn3d_prediction = "FAKE"

    cnn3d_confidence = 72.15


    # ----------------------------------------------
    # SAMPLE AI EXPLANATION
    # ----------------------------------------------

    ai_explanation = (
        "The Basic CNN produced a FAKE prediction "
        "with relatively high confidence. The model "
        "output was more strongly associated with the "
        "FAKE class based on visual features learned "
        "during training. The 3D CNN also produced a "
        "FAKE prediction, indicating that both models "
        "classified the provided content as FAKE."
    )


    # ----------------------------------------------
    # CREATE REPORT
    # ----------------------------------------------

    generator = (
        STREESHIELDCombinedReport()
    )


    report = generator.generate_report(
        cnn_prediction,
        cnn_confidence,
        cnn3d_prediction,
        cnn3d_confidence,
        ai_explanation
    )


    # ----------------------------------------------
    # DISPLAY REPORT
    # ----------------------------------------------

    print(report)


    print("\n13B TEST COMPLETE.")