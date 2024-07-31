import json
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EpipolarWarpOperator(nn.Module):
    def __init__(self):
        super(EpipolarWarpOperator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, source_intrinsics, target_intrinsics, source_pose, target_pose):
        dtype = x.dtype
        source_intrinsics = source_intrinsics.to(dtype)
        target_intrinsics = target_intrinsics.to(dtype)
        source_pose = source_pose.to(dtype)
        target_pose = target_pose.to(dtype)
        batch_size, channels, height, width = x.size()
        
        # Compute fundamental matrix for each sample in the batch
        F_batch = self.compute_fundamental_matrix_batch(source_intrinsics, target_intrinsics, source_pose, target_pose)
        
        # Generate pixel coordinates
        pixel_coords = self.generate_pixel_coords(batch_size, height, width, x.device)
        
        # Compute epipolar lines for each pixel in each sample
        epipolar_lines = torch.bmm(F_batch.transpose(1, 2), pixel_coords)
        
        # Sample features along epipolar lines
        sampled_features = self.sample_along_epipolar_lines(x, epipolar_lines, height, width)
        
        # Apply convolution and activation
        output = self.conv1(sampled_features)
        output = self.relu(output)
        
        return output

    def compute_fundamental_matrix_batch(self, source_intrinsics, target_intrinsics, source_pose, target_pose):
        # print("source_intrinsics",source_intrinsics)
        batch_size = source_intrinsics.shape[0]
        F_batch = []
        for i in range(batch_size):
            K_source = source_intrinsics[i]
            K_target = target_intrinsics[i]
            R_source = self.rotation_vector_to_matrix(source_pose[i, :3])
            t_source = source_pose[i, 3:].unsqueeze(1)
            R_target = self.rotation_vector_to_matrix(target_pose[i, :3])
            t_target = target_pose[i, 3:].unsqueeze(1)
            
            R_relative = torch.mm(R_source, R_target.t())
            t_relative = t_source - torch.mm(R_relative, t_target)
            
            E = torch.mm(self.skew_symmetric(t_relative.squeeze()), R_relative)
            F = torch.mm(torch.mm(torch.inverse(K_target).t(), E), torch.inverse(K_source))
            F_batch.append(F)
        
        return torch.stack(F_batch)

    def generate_pixel_coords(self, batch_size, height, width, device):
        x = torch.arange(width, device=device).float()
        y = torch.arange(height, device=device).float()
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x)
        pixel_coords = torch.stack((grid_x.flatten(), grid_y.flatten(), ones.flatten()), dim=0)
        pixel_coords = pixel_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        return pixel_coords

    def sample_along_epipolar_lines(self, x, epipolar_lines, height, width):
        batch_size, channels, _, _ = x.shape
        sampled_features = []
        y = torch.arange(height, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        y = y.expand(batch_size, 1, -1, 1)  # [batch_size, 1, height, 1]
        for i in range(width):
            for j in range(height):
                l = epipolar_lines[:, :, i*height + j].view(batch_size, 3, 1)
                a, b, c = l[:, 0], l[:, 1], l[:, 2]

                # 에피폴라 선과 이미지 경계의 교차점 계산
                x1 = torch.clamp(-c / (a + 1e-10), 0, width - 1)
                x2 = torch.clamp(-(b*(height-1) + c) / (a + 1e-10), 0, width - 1)
                y1 = torch.clamp(-c / (b + 1e-10), 0, height - 1)
                y2 = torch.clamp(-(a*(width-1) + c) / (b + 1e-10), 0, height - 1)

                # 에피폴라 선을 따라 일정 간격으로 샘플링 포인트 생성
                num_samples = 3  # 샘플링 포인트 수
                t = torch.linspace(0, 1, num_samples, device=x.device).view(1, -1).expand(batch_size, -1)
                sample_x = x1.view(-1, 1) * (1 - t) + x2.view(-1, 1) * t
                sample_y = y1.view(-1, 1) * (1 - t) + y2.view(-1, 1) * t

                # 정규화된 좌표로 변환
                grid = torch.stack((
                    2 * sample_x / (width - 1) - 1,
                    2 * sample_y / (height - 1) - 1
                ), dim=-1).view(batch_size, 1, -1, 2)

                # 샘플링 및 특징 추출
                sampled = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
                
                # 샘플링된 특징들의 평균 계산
                averaged_feature = sampled.mean(dim=3)  # [batch_size, channels, 1]
                sampled_features.append(averaged_feature.squeeze(2))

        # print("batch_size",batch_size,"channels",channels,"height",height,"width",width)
        return torch.stack(sampled_features, dim=2).view(batch_size, channels, height, width)

    @staticmethod
    def skew_symmetric(v):
        return torch.tensor([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]], device=v.device)

    @staticmethod
    def rotation_vector_to_matrix(rotation_vector):
        theta = torch.norm(rotation_vector)
        if theta < 1e-6:
            return torch.eye(3, device=rotation_vector.device)
        
        r = rotation_vector / theta
        I = torch.eye(3, device=rotation_vector.device)
        r_cross = torch.tensor([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ], device=rotation_vector.device)
        
        rotation_matrix = torch.cos(theta) * I + (1 - torch.cos(theta)) * torch.outer(r, r) + torch.sin(theta) * r_cross
        return rotation_matrix

def load_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def visualize_epipolar_geometry(operator, source_image, target_image, source_intrinsics, target_intrinsics, source_pose, target_pose, num_points=5, save_path=None):
    plt.figure(figsize=(20, 10))
    
    # Source image
    plt.subplot(1, 2, 1)
    source_img = source_image[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(source_img)
    plt.title('Source Image with Selected Points')
    plt.axis('off')
    
    # Target image
    plt.subplot(1, 2, 2)
    target_img = target_image[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(target_img)
    plt.title('Target Image with Epipolar Lines')
    plt.axis('off')
    
    # Compute Fundamental matrix
    F = operator.compute_fundamental_matrix_batch(source_intrinsics, target_intrinsics, source_pose, target_pose)[0]
    
    height, width = source_image.shape[2:]
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    
    for color in colors:
        # 소스 이미지에서 랜덤한 점 선택
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        point = torch.tensor([[x, y, 1]], dtype=torch.float32).t()
        
        # 소스 이미지에 선택된 점 표시
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'o', color=color, markersize=10)
        
        # 타겟 이미지의 Epipolar line 계산
        l = torch.matmul(F, point).squeeze()
        a, b, c = l.cpu().numpy()
        
        # 타겟 이미지에 Epipolar line 그리기
        plt.subplot(1, 2, 2)
        y_range = np.array([0, height-1])
        x_range = -(b*y_range + c) / (a + 1e-10)
        
        valid_mask = (x_range >= 0) & (x_range < width)
        x_range = x_range[valid_mask]
        y_range = y_range[valid_mask]
        
        if len(x_range) > 0 and len(y_range) > 0:
            plt.plot(x_range, y_range, color=color, linewidth=2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # Load annotations
    with open('{annotation 주소}', 'r') as f:
        annotations = json.load(f)

    # EpipolarWarpOperator 인스턴스 생성
    operator = EpipolarWarpOperator()

    for idx, annotation in enumerate(annotations):
        source_image_path = os.path.join('rawtest', annotation['source_image'])
        target_image_path = os.path.join('rawtest', annotation['target_image'])

        # 이미지 로드
        source_image = load_image(source_image_path)
        target_image = load_image(target_image_path)

        # 카메라 파라미터 설정
        source_intrinsics = torch.tensor(annotation['source_camera_intrinsic'], dtype=torch.float32).unsqueeze(0)
        target_intrinsics = torch.tensor(annotation['target_camera_intrinsic'], dtype=torch.float32).unsqueeze(0)
        source_pose = torch.tensor(annotation['source_camera_pose'], dtype=torch.float32).unsqueeze(0)
        target_pose = torch.tensor(annotation['target_camera_pose'], dtype=torch.float32).unsqueeze(0)

        # Epipolar line 시각화 및 저장
        save_dir = '{시각화 저장할 폴더 이름}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epipolar_correspondence_{idx}.png')
        
        visualize_epipolar_geometry(operator, source_image, target_image, source_intrinsics, target_intrinsics, source_pose, target_pose, save_path=save_path)

        print(f"Processed and saved visualization for image pair {idx + 1}")

if __name__ == "__main__":
    main()