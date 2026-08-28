import os
import shutil
import random
from collections import defaultdict


# ==================================================
# PATHS
# ==================================================

SOURCE_PATH = r"D:\STREESHIELD_VideoDataset\sequences"
OUTPUT_PATH = r"D:\STREESHIELD_VideoDataset\split_sequences"


# ==================================================
# SETTINGS
# ==================================================

CLASSES = ["real", "fake"]

SPLIT_RATIOS = {
    "train": 0.70,
    "valid": 0.15,
    "test": 0.15
}

SEED = 42

random.seed(SEED)


# ==================================================
# FIND VIDEOS WITH VALID SEQUENCES
# ==================================================

video_info = []

for class_name in CLASSES:

    class_path = os.path.join(
        SOURCE_PATH,
        class_name
    )

    if not os.path.exists(class_path):
        print(f"ERROR: Missing folder: {class_path}")
        raise SystemExit(1)

    for video_name in sorted(os.listdir(class_path)):

        video_path = os.path.join(
            class_path,
            video_name
        )

        if not os.path.isdir(video_path):
            continue

        sequence_files = [
            file
            for file in os.listdir(video_path)
            if file.lower().endswith(".npy")
        ]

        if len(sequence_files) == 0:
            continue

        video_info.append({
            "class": class_name,
            "video": video_name,
            "sequence_count": len(sequence_files)
        })


# ==================================================
# UNION-FIND FUNCTIONS
# Used to keep related REAL and FAKE videos
# in the same split.
# ==================================================

parent = {}


def find(node):
    """Find the root of a node."""

    if parent[node] != node:
        parent[node] = find(parent[node])

    return parent[node]


def union(a, b):
    """Connect two related nodes."""

    if a not in parent:
        parent[a] = a

    if b not in parent:
        parent[b] = b

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# ==================================================
# CREATE VIDEO NODES
# ==================================================

for item in video_info:

    node = f"{item['class']}:{item['video']}"

    parent[node] = node


# ==================================================
# CONNECT RELATED VIDEOS
#
# Example:
#   real:183
#   real:253
#   fake:183_253
#
# These belong to the same source group.
# ==================================================

for item in video_info:

    if item["class"] != "fake":
        continue

    fake_name = item["video"]

    parts = fake_name.split("_")

    if len(parts) != 2:
        continue

    source_a = f"real:{parts[0]}"
    source_b = f"real:{parts[1]}"
    fake_node = f"fake:{fake_name}"

    if source_a in parent:
        union(fake_node, source_a)

    if source_b in parent:
        union(fake_node, source_b)


# ==================================================
# CREATE CONNECTED COMPONENTS
# ==================================================

components = defaultdict(list)

for item in video_info:

    node = f"{item['class']}:{item['video']}"

    root = find(node)

    components[root].append(item)


component_list = []

for items in components.values():

    real_sequences = sum(
        item["sequence_count"]
        for item in items
        if item["class"] == "real"
    )

    fake_sequences = sum(
        item["sequence_count"]
        for item in items
        if item["class"] == "fake"
    )

    component_list.append({
        "items": items,
        "real": real_sequences,
        "fake": fake_sequences,
        "total": real_sequences + fake_sequences
    })


# ==================================================
# SHUFFLE COMPONENTS
# ==================================================

random.shuffle(component_list)


# ==================================================
# TOTAL COUNTS
# ==================================================

total_real = sum(
    component["real"]
    for component in component_list
)

total_fake = sum(
    component["fake"]
    for component in component_list
)

total_sequences = total_real + total_fake


# ==================================================
# TARGET COUNTS
# ==================================================

targets = {}

for split_name, ratio in SPLIT_RATIOS.items():

    targets[split_name] = {
        "real": total_real * ratio,
        "fake": total_fake * ratio,
        "total": total_sequences * ratio
    }


# ==================================================
# SORT COMPONENTS BY SIZE
# ==================================================

component_list.sort(
    key=lambda component: component["total"],
    reverse=True
)


# ==================================================
# SPLIT STORAGE
# ==================================================

splits = {
    "train": [],
    "valid": [],
    "test": []
}


current = {
    "train": {
        "real": 0,
        "fake": 0,
        "total": 0
    },
    "valid": {
        "real": 0,
        "fake": 0,
        "total": 0
    },
    "test": {
        "real": 0,
        "fake": 0,
        "total": 0
    }
}


# ==================================================
# ASSIGN COMPONENTS TO SPLITS
#
# Important:
# Train is given priority because it has
# the largest target.
# ==================================================

for component in component_list:

    best_split = None
    best_score = float("inf")

    for split_name in SPLIT_RATIOS:

        target_total = targets[
            split_name
        ]["total"]

        current_total = current[
            split_name
        ]["total"]

        # Proportion currently filled
        fill_ratio = (
            current_total / target_total
            if target_total > 0
            else float("inf")
        )

        # Penalize splits that are already full
        if fill_ratio < best_score:

            best_score = fill_ratio
            best_split = split_name

    splits[best_split].append(component)

    current[best_split]["real"] += (
        component["real"]
    )

    current[best_split]["fake"] += (
        component["fake"]
    )

    current[best_split]["total"] += (
        component["total"]
    )


# ==================================================
# CLEAN PREVIOUS OUTPUT
# ==================================================

if os.path.exists(OUTPUT_PATH):

    print("\nRemoving previous split folder...")

    shutil.rmtree(
        OUTPUT_PATH
    )


# ==================================================
# CREATE OUTPUT FOLDERS
# ==================================================

for split_name in SPLIT_RATIOS:

    for class_name in CLASSES:

        os.makedirs(
            os.path.join(
                OUTPUT_PATH,
                split_name,
                class_name
            ),
            exist_ok=True
        )


# ==================================================
# COPY SEQUENCES
# ==================================================

copied_sequences = 0

for split_name, components_in_split in splits.items():

    for component in components_in_split:

        # IMPORTANT:
        # component is a dictionary.
        # Actual video records are stored
        # inside component["items"].

        for item in component["items"]:

            class_name = item["class"]
            video_name = item["video"]

            source_folder = os.path.join(
                SOURCE_PATH,
                class_name,
                video_name
            )

            destination_folder = os.path.join(
                OUTPUT_PATH,
                split_name,
                class_name,
                video_name
            )

            os.makedirs(
                destination_folder,
                exist_ok=True
            )

            sequence_files = [
                file
                for file in os.listdir(
                    source_folder
                )
                if file.lower().endswith(".npy")
            ]

            for sequence_file in sequence_files:

                source_file = os.path.join(
                    source_folder,
                    sequence_file
                )

                destination_file = os.path.join(
                    destination_folder,
                    sequence_file
                )

                shutil.copy2(
                    source_file,
                    destination_file
                )

                copied_sequences += 1


# ==================================================
# VERIFY COPIED COUNTS
# ==================================================

verified_counts = {
    "train": {
        "real": 0,
        "fake": 0
    },
    "valid": {
        "real": 0,
        "fake": 0
    },
    "test": {
        "real": 0,
        "fake": 0
    }
}


for split_name in SPLIT_RATIOS:

    for class_name in CLASSES:

        class_path = os.path.join(
            OUTPUT_PATH,
            split_name,
            class_name
        )

        for root, _, files in os.walk(class_path):

            verified_counts[
                split_name
            ][class_name] += sum(
                1
                for file in files
                if file.lower().endswith(".npy")
            )


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========================================")
print("       VIDEO DATASET SPLIT SUMMARY")
print("========================================")

print("\nOriginal sequence totals:")
print("REAL :", total_real)
print("FAKE :", total_fake)
print("TOTAL:", total_sequences)


for split_name in SPLIT_RATIOS:

    real_count = verified_counts[
        split_name
    ]["real"]

    fake_count = verified_counts[
        split_name
    ]["fake"]

    total_count = (
        real_count + fake_count
    )

    print(f"\n{split_name.upper()}")

    print(
        "REAL sequences :",
        real_count
    )

    print(
        "FAKE sequences :",
        fake_count
    )

    print(
        "TOTAL sequences:",
        total_count
    )


print("\nTotal copied sequences:",
      copied_sequences)

print("Total verified sequences:",
      sum(
          verified_counts[split]["real"]
          + verified_counts[split]["fake"]
          for split in SPLIT_RATIOS
      ))

print("\nOutput path:")
print(OUTPUT_PATH)

print("\nTarget split:")
print("TRAIN = 70%")
print("VALID = 15%")
print("TEST  = 15%")

print("\nRandom seed:", SEED)

print("========================================")