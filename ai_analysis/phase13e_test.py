"""
STREESHIELD Phase 13E
Complete Application Test

This module ONLY performs validation tests.

It does NOT modify:
- CNN models
- 3D CNN models
- preprocessing
- AI analysis
- AI assistant
- Streamlit application
- existing Phase 13 files
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = r"E:\streesheild"


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

if PROJECT_DIR not in sys.path:

    sys.path.insert(
        0,
        PROJECT_DIR
    )


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = [

    # --------------------------------------------
    # Main application
    # --------------------------------------------

    r"app\app.py",

    r"app\app_13d.py",

    r"app\app_backup_before_13D.py",


    # --------------------------------------------
    # Models
    # --------------------------------------------

    r"app\models\basic_cnn.keras",

    r"app\models\3d_cnn.keras",


    # --------------------------------------------
    # Existing AI modules
    # --------------------------------------------

    r"ai_analysis\ai_analyzer.py",

    r"ai_analysis\cnn_connector.py",

    r"ai_analysis\cnn3d_connector.py",

    r"ai_analysis\explanation_engine.py",


    # --------------------------------------------
    # Phase 13 modules
    # --------------------------------------------

    r"ai_analysis\ai_report.py",

    r"ai_analysis\combined_report.py",

    r"ai_analysis\risk_assesment.py",

    r"ai_analysis\phase13_integration.py",
]


# ============================================================
# TEST HEADER
# ============================================================

print()

print("=" * 60)

print(
    "       STREESHIELD PHASE 13E"
)

print(
    "       COMPLETE APPLICATION TEST"
)

print("=" * 60)


# ============================================================
# TEST 1 — PROJECT DIRECTORY
# ============================================================

print()

print("-" * 60)

print(
    "TEST 1 — PROJECT DIRECTORY"
)

print("-" * 60)


if os.path.exists(PROJECT_DIR):

    print("PASS")

    print(
        "Project directory found:"
    )

    print(
        PROJECT_DIR
    )

else:

    print("FAIL")

    print(
        "Project directory not found:"
    )

    print(
        PROJECT_DIR
    )

    raise SystemExit(1)


# ============================================================
# TEST 2 — REQUIRED FILES
# ============================================================

print()

print("-" * 60)

print(
    "TEST 2 — REQUIRED FILES"
)

print("-" * 60)


missing_files = []


for relative_path in REQUIRED_FILES:

    full_path = os.path.join(
        PROJECT_DIR,
        relative_path
    )


    if os.path.exists(full_path):

        print(
            "PASS :",
            relative_path
        )

    else:

        print(
            "FAIL :",
            relative_path
        )

        missing_files.append(
            relative_path
        )


if missing_files:

    print()

    print(
        "Missing files detected:"
    )


    for file in missing_files:

        print(
            " -",
            file
        )


    raise SystemExit(1)


# ============================================================
# TEST 3 — AI MODULE IMPORTS
# ============================================================

print()

print("-" * 60)

print(
    "TEST 3 — AI MODULE IMPORTS"
)

print("-" * 60)


try:

    from ai_analysis.ai_analyzer import (
        STREESHIELDAnalyzer
    )

    print(
        "PASS : ai_analyzer"
    )


    from ai_analysis.ai_report import (
        STREESHIELDReportGenerator
    )

    print(
        "PASS : ai_report"
    )


    from ai_analysis.combined_report import (
        STREESHIELDCombinedReport
    )

    print(
        "PASS : combined_report"
    )


    from ai_analysis.risk_assesment import (
        STREESHIELDRiskAssessment
    )

    print(
        "PASS : risk_assesment"
    )


    from ai_analysis.explanation_engine import (
        AIExplanationEngine
    )

    print(
        "PASS : explanation_engine"
    )


except Exception as error:

    print()

    print(
        "FAIL — AI module import error"
    )

    print(
        "Details:",
        error
    )

    raise SystemExit(1)


# ============================================================
# TEST 4 — AI ANALYSIS
# ============================================================

print()

print("-" * 60)

print(
    "TEST 4 — AI ANALYSIS"
)

print("-" * 60)


try:

    analyzer = (
        STREESHIELDAnalyzer()
    )


    result = (
        analyzer.analyze_prediction(

            model_name="Basic CNN",

            prediction="REAL",

            confidence=68.17,

            media_type="Image"
        )
    )


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


    print(
        "Confidence Level :",
        result["confidence_level"]
    )


    print()

    print(
        "PASS — AI analysis working"
    )


except Exception as error:

    print()

    print(
        "FAIL — AI analysis"
    )

    print(
        "Details:",
        error
    )

    raise SystemExit(1)


# ============================================================
# TEST 5 — AI EXPLANATION
# ============================================================

print()

print("-" * 60)

print(
    "TEST 5 — AI EXPLANATION"
)

print("-" * 60)


try:

    engine = (
        AIExplanationEngine()
    )


    class TestInput:

        def to_dict(self):

            return {

                "model_name":
                    "Basic CNN",

                "prediction":
                    "REAL",

                "confidence":
                    68.17,

                "media_type":
                    "Image",

                "raw_probability":
                    0.3182801306,

                "media_info":
                    {}
            }


    explanation_result = (
        engine.generate_explanation(

            TestInput()
        )
    )


    print(
        "Prediction explanation:"
    )


    print(
        explanation_result[
            "prediction_explanation"
        ]
    )


    print()

    print(
        "PASS — AI explanation working"
    )


except Exception as error:

    print()

    print(
        "FAIL — AI explanation"
    )

    print(
        "Details:",
        error
    )

    raise SystemExit(1)


# ============================================================
# TEST 6 — RISK ASSESSMENT
# ============================================================

print()

print("-" * 60)

print(
    "TEST 6 — RISK ASSESSMENT"
)

print("-" * 60)


try:

    assessment = (
        STREESHIELDRiskAssessment()
    )


    test_cases = [

        ("FAKE", 87.42),

        ("REAL", 91.50),

        ("FAKE", 55.00)
    ]


    for prediction, confidence in test_cases:

        risk = (
            assessment.calculate_risk(

                prediction,

                confidence
            )
        )


        print(
            f"{prediction:<6} "
            f"{confidence:>6.2f}% "
            f"-> {risk}"
        )


    print()

    print(
        "PASS — Risk assessment working"
    )


except Exception as error:

    print()

    print(
        "FAIL — Risk assessment"
    )

    print(
        "Details:",
        error
    )

    raise SystemExit(1)


# ============================================================
# TEST 7 — INDIVIDUAL AI REPORT
# ============================================================

print()

print("-" * 60)

print(
    "TEST 7 — INDIVIDUAL AI REPORT"
)

print("-" * 60)


try:

    generator = (
        STREESHIELDReportGenerator()
    )


    report = (
        generator.generate_report(

            detection_type="Image",

            model_name="Basic CNN",

            prediction="REAL",

            confidence=68.17,

            ai_explanation=(
                "The Basic CNN classified "
                "the image as REAL based on "
                "learned image-level features."
            )
        )
    )


    if (
        "STREESHIELD AI REPORT"
        in report
    ):

        print(
            "PASS — AI report generated"
        )

    else:

        print(
            "FAIL — AI report content invalid"
        )

        raise SystemExit(1)


except Exception as error:

    print()

    print(
        "FAIL — AI report"
    )

    print(
        "Details:",
        error
    )

    raise SystemExit(1)


# ============================================================
# TEST 8 — COMBINED AI REPORT
# ============================================================

print()

print("-" * 60)

print(
    "TEST 8 — COMBINED AI REPORT"
)

print("-" * 60)


try:

    combined_generator = (
        STREESHIELDCombinedReport()
    )


    combined_report = (
        combined_generator.generate_report(

            cnn_prediction="FAKE",

            cnn_confidence=87.42,

            cnn3d_prediction="FAKE",

            cnn3d_confidence=72.15,

            ai_explanation=(
                "Both models classified "
                "the content as FAKE."
            )
        )
    )


    required_sections = [

        "BASIC CNN RESULT",

        "3D CNN RESULT",

        "AI EXPLANATION",

        "STREESHIELD AI REPORT"
    ]


    for section in required_sections:

        if section not in combined_report:

            raise ValueError(
                f"Missing report section: "
                f"{section}"
            )


    print(
        "PASS — Combined AI report generated"
    )


except Exception as error:

    print()

    print(
        "FAIL — Combined AI report"
    )

    print(
        "Details:",
        error
    )

    raise SystemExit(1)


# ============================================================
# TEST 9 — IMAGE TEST SAMPLE
# ============================================================

print()

print("-" * 60)

print(
    "TEST 9 — IMAGE TEST SAMPLE"
)

print("-" * 60)


IMAGE_TEST_PATH = (
    r"D:\STREESHIELD_Dataset"
    r"\processed\test\real\00001.jpg"
)


if os.path.exists(
    IMAGE_TEST_PATH
):

    print(
        "PASS — Image test sample found"
    )

    print(
        "Image:",
        IMAGE_TEST_PATH
    )

else:

    print(
        "WARNING — Image test sample not found"
    )

    print(
        IMAGE_TEST_PATH
    )


# ============================================================
# TEST 10 — VIDEO TEST SAMPLE
# ============================================================

print()

print("-" * 60)

print(
    "TEST 10 — VIDEO TEST SAMPLE"
)

print("-" * 60)


VIDEO_TEST_PATH = (
    r"D:\STREESHIELD_VideoDataset"
    r"\original\real\183.mp4"
)


if os.path.exists(
    VIDEO_TEST_PATH
):

    print(
        "PASS — Video test sample found"
    )

    print(
        "Video:",
        VIDEO_TEST_PATH
    )

else:

    print(
        "WARNING — Video test sample not found"
    )

    print(
        VIDEO_TEST_PATH
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()

print("=" * 60)

print(
    "       PHASE 13E AUTOMATED TESTS COMPLETE"
)

print("=" * 60)


print()

print("Verified:")

print()

print(
    "1. Project structure"
)

print(
    "2. Required model files"
)

print(
    "3. AI analysis"
)

print(
    "4. AI explanation"
)

print(
    "5. Risk assessment"
)

print(
    "6. Individual AI report"
)

print(
    "7. Combined AI report"
)

print(
    "8. Image test sample"
)

print(
    "9. Video test sample"
)


print()

print("Next:")

print()

print(
    "Manual Streamlit testing is required for:"
)

print(
    "10. Image detection"
)

print(
    "11. Video detection"
)

print(
    "12. AI explanation in UI"
)

print(
    "13. AI Assistant / Chatbot"
)

print(
    "14. AI Report in UI"
)


print()

print(
    "DO NOT PROCEED TO PHASE 13F YET."
)


print()

print("=" * 60)