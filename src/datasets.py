import numpy as np
from torchvision.io import decode_image
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(
        self,
        input_img_dir,
        output_img_dir,
        num_instances,
        num_frames_per_instance,
        transform=None,
        target_transform=None,
    ):
        self.input_img_dir = input_img_dir
        self.output_img_dir = output_img_dir
        self.num_instances = num_instances
        self.num_frames_per_instance = num_frames_per_instance
        self.num_frames = self.num_instances * self.num_frames_per_instance
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        instance = str(idx // self.num_frames_per_instance).zfill(4)
        frame = str(idx % self.num_frames_per_instance).zfill(4) + '.png'
        print(instance, frame)

        input_img_path = self.input_img_dir / instance / frame
        output_img_path = self.output_img_dir / instance / frame

        input_img = decode_image(input_img_path.resolve())
        output_img = decode_image(output_img_path.resolve())

        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            output_img = self.target_transform(output_img)

        return input_img, output_img
