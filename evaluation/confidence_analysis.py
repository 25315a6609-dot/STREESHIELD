import os
import pandas as pd
import numpy as np


# ==================================================
# PATHS
# ==================================================

BASIC_PATH = (
    r"E:\streesheild\evaluation\test_predictions.csv"
)

CNN3D_PATH = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\confidence_analysis.txt"
)


# ==================================================
# ANALYZE ONE MODEL
# ==================================================

def analyze_model(
    path,
    model_name
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    data = pd.read_csv(path)

    probability = pd.to_numeric(
        data["predicted_probability"],
        errors="coerce"
    )

    actual = data["actual_label"].str.upper()

    predicted = data["predicted_label"].str.upper()

    # ----------------------------------------------
    # CONFIDENCE FOR PREDICTED CLASS
    # ----------------------------------------------

    confidence = np.where(
        probability >= 0.5,
        probability,
        1 - probability
    )

    confidence = confidence * 100

    # ----------------------------------------------
    # CORRECT / INCORRECT
    # ----------------------------------------------

    correct = (
        actual == predicted
    )

    # ----------------------------------------------
    # REAL PREDICTIONS
    # ----------------------------------------------

    real_mask = predicted == "REAL"

    fake_mask = predicted == "FAKE"

    # ----------------------------------------------
    # STATISTICS
    # ----------------------------------------------

    result = {

        "model": model_name,

        "total_predictions": len(data),

        "real_predictions": int(
            real_mask.sum()
        ),

        "fake_predictions": int(
            fake_mask.sum()
        ),

        "average_confidence": float(
            confidence.mean()
        ),

        "minimum_confidence": float(
            confidence.min()
        ),

        "maximum_confidence": float(
            confidence.max()
        ),

        "average_real_confidence": (
            float(confidence[real_mask].mean())
            if real_mask.any()
            else 0.0
        ),

        "average_fake_confidence": (
            float(confidence[fake_mask].mean())
            if fake_mask.any()
            else 0.0
        ),

        "average_correct_confidence": (
            float(confidence[correct].mean())
            if correct.any()
            else 0.0
        ),

        "average_incorrect_confidence": (
            float(confidence[~correct].mean())
            if (~correct).any()
            else 0.0
        ),

        "correct_predictions": int(
            correct.sum()
        ),

        "incorrect_predictions": int(
            (~correct).sum()
        )
    }

    return result


# ==================================================
# ANALYZE MODELS
# ==================================================

basic = analyze_model(
    BASIC_PATH,
    "Basic CNN"
)

cnn3d = analyze_model(
    CNN3D_PATH,
    "3D CNN"
)


# ==================================================
# DISPLAY
# ==================================================

print("\n========================================")
print("     PHASE 10D — CONFIDENCE ANALYSIS")
print("========================================")


for result in [basic, cnn3d]:

    print(
        f"\n========== {result['model']} =========="
    )

    print(
        "Total predictions       :",
        result["total_predictions"]
    )

    print(
        "REAL predictions        :",
        result["real_predictions"]
    )

    print(
        "FAKE predictions        :",
        result["fake_predictions"]
    )

    print(
        "Average confidence     :",
        f"{result['average_confidence']:.2f}%"
    )

    print(
        "Minimum confidence     :",
        f"{result['minimum_confidence']:.2f}%"
    )

    print(
        "Maximum confidence     :",
        f"{result['maximum_confidence']:.2f}%"
    )

    print(
        "Average REAL confidence:",
        f"{result['average_real_confidence']:.2f}%"
    )

    print(
        "Average FAKE confidence:",
        f"{result['average_fake_confidence']:.2f}%"
    )

    print(
        "Correct prediction confidence:",
        f"{result['average_correct_confidence']:.2f}%"
    )

    print(
        "Incorrect prediction confidence:",
        f"{result['average_incorrect_confidence']:.2f}%"
    )


# ==================================================
# INTERPRETATION
# ==================================================

print("\n========================================")
print("          INTERPRETATION")
print("========================================")

print(
    "\nBasic CNN:"
)

print(
    "- Produces both REAL and FAKE predictions."
)

print(
    "- Confidence can therefore be analyzed "
    "across both classes."
)

print(
    "\n3D CNN:"
)

print(
    "- Predicts every test sequence as FAKE."
)

print(
    "- Its confidence is expected to be close "
    "to 50%, indicating weak certainty."
)


# ==================================================
# SAVE REPORT
# ==================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    for result in [basic, cnn3d]:

        file.write(
            f"{result['model']}\n"
        )

        file.write(
            f"Total predictions: "
            f"{result['total_predictions']}\n"
        )

        file.write(
            f"REAL predictions: "
            f"{result['real_predictions']}\n"
        )

        file.write(
            f"FAKE predictions: "
            f"{result['fake_predictions']}\n"
        )

        file.write(
            f"Average confidence: "
            f"{result['average_confidence']:.2f}%\n"
        )

        file.write(
            f"Minimum confidence: "
            f"{result['minimum_confidence']:.2f}%\n"
        )

        file.write(
            f"Maximum confidence: "
            f"{result['maximum_confidence']:.2f}%\n"
        )

        file.write(
            f"Average REAL confidence: "
            f"{result['average_real_confidence']:.2f}%\n"
        )

        file.write(
            f"Average FAKE confidence: "
            f"{result['average_fake_confidence']:.2f}%\n"
        )

        file.write(
            f"Correct prediction confidence: "
            f"{result['average_correct_confidence']:.2f}%\n"
        )

        file.write(
            f"Incorrect prediction confidence: "
            f"{result['average_incorrect_confidence']:.2f}%\n"
        )

        file.write(
            f"Correct predictions: "
            f"{result['correct_predictions']}\n"
        )

        file.write(
            f"Incorrect predictions: "
            f"{result['incorrect_predictions']}\n"
        )

        file.write("\n")


print("\nConfidence analysis saved to:")
print(OUTPUT_PATH)

print("\n========================================")
print("       PHASE 10D COMPLETED")
print("========================================")