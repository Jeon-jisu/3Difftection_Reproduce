import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_camera_path(original_traj_data, updated_traj_data, title, output_file):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 데이터의 1/4만 사용
    sample_rate = 4
    original_positions = np.array([entry[4:7] for entry in original_traj_data[::sample_rate]])
    updated_positions = np.array([entry[4:7] for entry in updated_traj_data[::sample_rate]])
    
    # 원본 경로
    ax.plot(original_positions[:, 0], original_positions[:, 1], original_positions[:, 2], 'b-', label='Original Path')
    
    # 업데이트된 경로
    ax.plot(updated_positions[:, 0], updated_positions[:, 1], updated_positions[:, 2], 'r-', label='Updated Path')
    
    # 카메라 방향 표시 (10개의 프레임에 대해서만)
    num_arrows = 10
    step = len(original_positions) // num_arrows
    for i in range(0, len(original_positions), step):
        # 원본 카메라 방향
        orig_rot = R.from_rotvec(original_traj_data[i*sample_rate][1:4])
        orig_dir = orig_rot.apply([0, 0, 1])  # 카메라는 z축 방향을 바라본다고 가정
        ax.quiver(original_positions[i, 0], original_positions[i, 1], original_positions[i, 2],
                  orig_dir[0], orig_dir[1], orig_dir[2], color='b', length=0.1, normalize=True)
        
        # 업데이트된 카메라 방향
        updated_rot = R.from_rotvec(updated_traj_data[i*sample_rate][1:4])
        updated_dir = updated_rot.apply([0, 0, 1])
        ax.quiver(updated_positions[i, 0], updated_positions[i, 1], updated_positions[i, 2],
                  updated_dir[0], updated_dir[1], updated_dir[2], color='r', length=0.1, normalize=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    
    # 그래프를 파일로 저장
    plt.savefig(output_file)
    plt.close(fig) 
# Define the rotation matrices for sky_direction
def get_rotation_matrix(sky_direction):
    if sky_direction.lower() == 'left':
        angle = -np.pi / 2  # 270 degrees counter-clockwise
    elif sky_direction.lower() == 'right':
        angle = np.pi / 2  # 90 degrees clockwise
    elif sky_direction.lower() == 'down':
        angle = np.pi  # 180 degrees
    else:
        angle = 0  # No rotation needed for 'up'
    return R.from_euler('z', angle).as_matrix()

# Convert axis-angle to rotation matrix
def axis_angle_to_rotation_matrix(axis_angle):
    rotation_vector = np.array(axis_angle)
    rotation = R.from_rotvec(rotation_vector)
    return rotation.as_matrix()

# Convert rotation matrix to axis-angle
def rotation_matrix_to_axis_angle(rotation_matrix):
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_rotvec()

# Apply rotation to the translation vector
def apply_rotation_to_translation(rotation_matrix, translation):
    return np.dot(rotation_matrix, translation)

# Update the trajectory data
def update_traj_data(traj_data, sky_direction):
    rotation_matrix_sky = get_rotation_matrix(sky_direction)
    updated_traj_data = []
    
    for entry in traj_data:
        timestamp = entry[0]
        axis_angle = entry[1:4]
        translation = entry[4:7]
        
        # Convert axis-angle to rotation matrix
        rotation_matrix = axis_angle_to_rotation_matrix(axis_angle)
        
        # Apply sky_direction rotation
        new_rotation_matrix = np.dot(rotation_matrix_sky, rotation_matrix)
        
        # Convert back to axis-angle
        new_axis_angle = rotation_matrix_to_axis_angle(new_rotation_matrix)
        
        # Update translation vector
        new_translation = apply_rotation_to_translation(rotation_matrix_sky, translation)
        
        updated_entry = [timestamp] + list(new_axis_angle) + list(new_translation)
        updated_traj_data.append(updated_entry)
    
    return updated_traj_data

# Read traj data from file
def read_traj_file(file_path):
    traj_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split()))
            traj_data.append(parts)
    return traj_data

# Write updated traj data to file
def write_traj_file(file_path, traj_data):
    with open(file_path, 'w') as file:
        for entry in traj_data:
            line = ' '.join(map(str, entry))
            file.write(line + '\n')

# Read metadata.csv file
def read_metadata(metadata_file):
    metadata = {}
    with open(metadata_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = row['video_id']
            sky_direction = row['sky_direction']
            metadata[video_id] = sky_direction
    return metadata

# Main function
def main(base_folder, metadata_file):
    # Read the metadata
    metadata = read_metadata(metadata_file)
    
    # List all subfolders in the base_folder
    subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]
    
    missing_files = 0
    processed_files = 0
    
    # Process each subfolder
    for folder_name in subfolders:
        input_file = os.path.join(base_folder, folder_name, 'lowres_wide.traj')
        output_file = os.path.join(base_folder, folder_name, 'lowres_wide_after_process.traj')
        
        # Get the sky_direction for the given folder_name
        sky_direction = metadata.get(folder_name, 'up')  # Default to 'up' if not found
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            missing_files += 1
            continue
        
        # Read the original traj data
        traj_data = read_traj_file(input_file)
        
        # Update the traj data
        updated_traj_data = update_traj_data(traj_data, sky_direction)
        
        # Write the updated traj data to a new file
        write_traj_file(output_file, updated_traj_data)
        
        # 시각화 추가
        # vis_output_file = os.path.join(base_folder, folder_name, f'camera_path_visualization_{folder_name}.png')
        # visualize_camera_path(traj_data, updated_traj_data, 
        #                       f"Camera Path for {folder_name} (Sky Direction: {sky_direction})",
        #                       vis_output_file)
        # print(f"Visualization saved for {folder_name} at {vis_output_file}")
        
        processed_files += 1
    
    print(f"\nProcessing complete.")
    print(f"Total folders: {len(subfolders)}")
    print(f"Processed files: {processed_files}")
    print(f"Missing files: {missing_files}")

# Example usage
base_folder = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/Training'
metadata_file = '/node_data/urp24s_jsjeon/3Difftection_Reproduce/ControlNet2/raw/metadata.csv'

main(base_folder, metadata_file)
