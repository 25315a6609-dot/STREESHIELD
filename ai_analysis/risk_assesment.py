"""
STREESHIELD Phase 13C
Risk Assessment Module

Provides a simple project-level risk classification:

LOW
MEDIUM
HIGH

The risk level is based on:
- Prediction
- Model confidence

This module does not modify any existing
STREESHIELD model or detection files.
"""


# ==================================================
# RISK ASSESSMENT CLASS
# ==================================================

class STREESHIELDRiskAssessment:

    def __init__(self):

        self.project_name = "STREESHIELD"


    # ==================================================
    # CALCULATE RISK
    # ==================================================

    def calculate_risk(
        self,
        prediction,
        confidence
    ):

        prediction = str(
            prediction
        ).upper().strip()

        confidence = float(
            confidence
        )


        # ----------------------------------------------
        # VALIDATE CONFIDENCE
        # ----------------------------------------------

        if confidence < 0 or confidence > 100:

            raise ValueError(
                "Confidence must be between 0 and 100."
            )


        # ----------------------------------------------
        # VALIDATE PREDICTION
        # ----------------------------------------------

        if prediction not in [
            "REAL",
            "FAKE"
        ]:

            raise ValueError(
                "Prediction must be REAL or FAKE."
            )


        # ----------------------------------------------
        # RISK LOGIC
        # ----------------------------------------------

        if prediction == "FAKE":

            if confidence >= 70:

                risk = "HIGH"

            else:

                risk = "MEDIUM"


        else:

            if confidence >= 70:

                risk = "LOW"

            else:

                risk = "MEDIUM"


        return risk


    # ==================================================
    # RISK DESCRIPTION
    # ==================================================

    def get_description(
        self,
        risk
    ):

        risk = str(
            risk
        ).upper().strip()


        if risk == "LOW":

            return (
                "The model indicates a lower risk of "
                "deepfake based on the current prediction "
                "and confidence."
            )


        elif risk == "MEDIUM":

            return (
                "The result has some uncertainty. "
                "Additional analysis or evidence is "
                "recommended before making a conclusion."
            )


        elif risk == "HIGH":

            return (
                "The model indicates a higher risk of "
                "deepfake based on the current prediction "
                "and confidence. Additional evidence is "
                "recommended before treating the result "
                "as definitive."
            )


        else:

            raise ValueError(
                "Risk must be LOW, MEDIUM, or HIGH."
            )


# ==================================================
# 13C TEST
# ==================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       STREESHIELD PHASE 13C")
    print("       RISK ASSESSMENT TEST")
    print("========================================")


    assessment = (
        STREESHIELDRiskAssessment()
    )


    # ----------------------------------------------
    # TEST 1
    # ----------------------------------------------

    prediction = "FAKE"
    confidence = 87.42

    risk = assessment.calculate_risk(
        prediction,
        confidence
    )

    description = assessment.get_description(
        risk
    )


    print("\n----------------------------------------")
    print("TEST 1")
    print("----------------------------------------")

    print(
        "Prediction :",
        prediction
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print(
        "Risk       :",
        risk
    )

    print(
        "Description:",
        description
    )


    # ----------------------------------------------
    # TEST 2
    # ----------------------------------------------

    prediction = "REAL"
    confidence = 91.50

    risk = assessment.calculate_risk(
        prediction,
        confidence
    )

    description = assessment.get_description(
        risk
    )


    print("\n----------------------------------------")
    print("TEST 2")
    print("----------------------------------------")

    print(
        "Prediction :",
        prediction
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print(
        "Risk       :",
        risk
    )

    print(
        "Description:",
        description
    )


    # ----------------------------------------------
    # TEST 3
    # ----------------------------------------------

    prediction = "FAKE"
    confidence = 55.00

    risk = assessment.calculate_risk(
        prediction,
        confidence
    )

    description = assessment.get_description(
        risk
    )


    print("\n----------------------------------------")
    print("TEST 3")
    print("----------------------------------------")

    print(
        "Prediction :",
        prediction
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print(
        "Risk       :",
        risk
    )

    print(
        "Description:",
        description
    )


    # ----------------------------------------------
    # COMPLETION
    # ----------------------------------------------

    print("\n========================================")
    print("        13C TEST COMPLETE")
    print("========================================")