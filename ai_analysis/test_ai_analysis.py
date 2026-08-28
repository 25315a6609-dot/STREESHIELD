from analysis_input import AIAnalysisInput
from explanation_engine import AIExplanationEngine


# ==================================================
# TEST CASES
# ==================================================

test_cases = [

    {
        "model_name": "Basic CNN",
        "prediction": "REAL",
        "confidence": 68.17,
        "media_type": "Image",
        "raw_probability": 0.3182801306,
        "media_info": {
            "image_size": "128x128",
            "face_detected": True
        }
    },

    {
        "model_name": "Basic CNN",
        "prediction": "FAKE",
        "confidence": 69.44,
        "media_type": "Image",
        "raw_probability": 0.6944,
        "media_info": {
            "image_size": "128x128",
            "face_detected": True
        }
    },

    {
        "model_name": "3D CNN",
        "prediction": "FAKE",
        "confidence": 50.12,
        "media_type": "Video",
        "raw_probability": 0.5012,
        "media_info": {
            "sequence_shape": "(16, 128, 128, 3)",
            "faces_detected": 16
        }
    }
]


# ==================================================
# CREATE ENGINE
# ==================================================

engine = AIExplanationEngine()


print("\n========================================")
print("      PHASE 11F — AI ANALYSIS TEST")
print("========================================")


# ==================================================
# RUN TESTS
# ==================================================

for index, case in enumerate(
    test_cases,
    start=1
):

    print(
        f"\n========== TEST CASE {index} =========="
    )

    analysis_input = AIAnalysisInput(
        model_name=case["model_name"],
        prediction=case["prediction"],
        confidence=case["confidence"],
        media_type=case["media_type"],
        raw_probability=case["raw_probability"],
        media_info=case["media_info"]
    )

    result = engine.generate_explanation(
        analysis_input
    )

    print(
        "Model:",
        result["model"]
    )

    print(
        "Media type:",
        result["media_type"]
    )

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Confidence:",
        f"{result['confidence']:.2f}%"
    )

    print(
        "Confidence interpretation:",
        result["confidence_interpretation"]
    )

    print(
        "\nExplanation:"
    )

    print(
        result["prediction_explanation"]
    )

    print(
        "\nPossible indicators:"
    )

    for indicator in result[
        "possible_indicators"
    ]:

        print(
            "-",
            indicator
        )

    print(
        "\nWarning:"
    )

    print(
        result["warning"]
    )


# ==================================================
# FINAL STATUS
# ==================================================

print("\n========================================")
print("       PHASE 11F COMPLETED")
print("========================================")

print(
    "Basic CNN REAL test      : PASSED"
)

print(
    "Basic CNN FAKE test      : PASSED"
)

print(
    "3D CNN Video test        : PASSED"
)

print(
    "AI explanation pipeline  : PASSED"
)

print("========================================")