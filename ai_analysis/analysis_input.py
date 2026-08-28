"""
STREESHIELD Phase 11D
Unified AI analysis input.
"""


class AIAnalysisInput:

    def __init__(
        self,
        model_name,
        prediction,
        confidence,
        media_type,
        raw_probability=None,
        media_info=None
    ):

        self.model_name = str(
            model_name
        )

        self.prediction = str(
            prediction
        ).upper()

        self.confidence = float(
            confidence
        )

        self.media_type = str(
            media_type
        )

        self.raw_probability = (
            None
            if raw_probability is None
            else float(raw_probability)
        )

        self.media_info = (
            {}
            if media_info is None
            else media_info
        )


    # ==================================================
    # VALIDATE
    # ==================================================

    def validate(self):

        if self.prediction not in [
            "REAL",
            "FAKE"
        ]:

            raise ValueError(
                "Prediction must be REAL or FAKE."
            )

        if not 0 <= self.confidence <= 100:

            raise ValueError(
                "Confidence must be between 0 and 100."
            )

        if self.raw_probability is not None:

            if not 0 <= self.raw_probability <= 1:

                raise ValueError(
                    "Raw probability must be between 0 and 1."
                )

        return True


    # ==================================================
    # CONVERT TO DICTIONARY
    # ==================================================

    def to_dict(self):

        self.validate()

        return {

            "model_name":
                self.model_name,

            "prediction":
                self.prediction,

            "confidence":
                self.confidence,

            "media_type":
                self.media_type,

            "raw_probability":
                self.raw_probability,

            "media_info":
                self.media_info
        }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

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

    print("\n========================================")
    print("       PHASE 11D INPUT TEST")
    print("========================================")

    print(
        analysis_input.to_dict()
    )

    print("\nValidation:")
    print(
        analysis_input.validate()
    )

    print("\n========================================")
    print("          11D TEST COMPLETE")
    print("========================================")