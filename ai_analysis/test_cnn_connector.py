from cnn_connector import BasicCNNConnector


MODEL_PATH = (
    r"E:\streesheild\models\trained_basic_cnn.keras"
)


print("\n========================================")
print("       BASIC CNN CONNECTOR TEST")
print("========================================")


connector = BasicCNNConnector(
    MODEL_PATH
)


# Example output from the existing CNN
result = connector.analyze_output(
    0.3182801306
)


print("\nModel:")
print(result["model"])

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
print("          11B TEST COMPLETE")
print("========================================")