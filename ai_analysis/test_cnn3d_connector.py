from cnn3d_connector import CNN3DConnector


MODEL_PATH = (
    r"E:\streesheild\models\trained_3d_cnn.keras"
)


print("\n========================================")
print("       3D CNN CONNECTOR TEST")
print("========================================")


connector = CNN3DConnector(
    MODEL_PATH
)


# Test probability from the existing
# 3D CNN prediction behavior.
result = connector.analyze_output(
    0.5012
)


print("\nModel:")
print(
    result["model"]
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
    "Raw probability:",
    result["raw_probability"]
)


print("\n========================================")
print("          11C TEST COMPLETE")
print("========================================")