import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load JSON file
def load_json(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data

# Convert rotation vector to rotation matrix
def rotation_vector_to_matrix(rotation_vector):
    return R.from_rotvec(rotation_vector).as_matrix()

# Calculate relative rotation angle (degrees) between two camera poses
def calculate_relative_angle(source_pose, target_pose):
    source_rotation = source_pose[3:6]
    target_rotation = target_pose[3:6]

    # Convert to rotation matrices
    R1 = rotation_vector_to_matrix(source_rotation)
    R2 = rotation_vector_to_matrix(target_rotation)

    # Relative rotation matrix and angle
    R_rel = R2 @ R1.T
    angle_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    return np.degrees(angle_rad)

# Main function
def main(json_path, output_path):
    data = load_json(json_path)
    angles = []

    # Calculate relative angles
    for pair in data:
        source_pose = pair["source_camera_pose"]
        target_pose = pair["target_camera_pose"]
        relative_angle = calculate_relative_angle(source_pose, target_pose)
        angles.append(relative_angle)

    # Calculate histogram with 10-degree bins
    bins = np.arange(0, 181, 10)
    hist, _ = np.histogram(angles, bins=bins)

    # Print frequencies for 10-degree bins
    print("Frequency per 10-degree bin:")
    for i in range(len(hist)):
        print(f"{bins[i]} - {bins[i+1]} degrees: {hist[i]}")

    # Plot histogram (with changes: remove title, enlarge labels, rename Y-axis)
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], hist, width=10, color='royalblue', edgecolor='black', align='edge')
    plt.xlabel("Relative Rotation Angle (degrees)", fontsize=14)
    plt.ylabel("Number of samples", fontsize=14)
    plt.xticks(np.arange(0, 181, 20), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save the histogram as PDF
    output_pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to {output_pdf_path}")


# Path to JSON file and output image
json_path = "/node_data_2/urp24s_jsjeon/3Difftection_Reproduce/omni3d/controlnet/raw/annotations.json"
output_path = "relative_rotation_angles_histogram.png"

main(json_path, output_path)
