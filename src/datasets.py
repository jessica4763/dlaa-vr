import torch
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

    @staticmethod
    def warp(input_image: torch.Tensor) -> torch.Tensor:
        # TODO: Jitter compensation.
        # TODO: Warp without motion vectors; choose something suitable
        # for a forward rendering pipeline.
        return input_image

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        instance = str(idx // self.num_frames_per_instance).zfill(4)

        curr_frame_num = idx % self.num_frames_per_instance
        curr_frame = str(curr_frame_num).zfill(4) + '.png'
        curr_input_img_path = self.input_img_dir / instance / curr_frame
        curr_output_img_path = self.output_img_dir / instance / curr_frame
        curr_input_img = decode_image(curr_input_img_path.resolve())
        curr_output_img = decode_image(curr_output_img_path.resolve())

        # Teacher forcing: use the gound truth as the previous frame during
        # training. Limitation is that the network never explicitly learns
        # to use its own output as the previous frame during inference.

        # Alternatively, we could gradually introduce using the model's own
        # output as the previous frame during inference.

        # When there is no previous frame, i.e. this is the first frame,
        # duplicate the current frame and use it as the previous frame.
        prev_frame_num = 0 if curr_frame_num == 0 else curr_frame_num - 1
        prev_frame = str(prev_frame_num).zfill(4) + '.png'
        prev_output_img_path = self.output_img_dir / instance / prev_frame
        prev_output_img = decode_image(prev_output_img_path.resolve())

        if self.transform:
            prev_output_img = self.transform(prev_output_img)
            curr_input_img = self.transform(curr_input_img)
        if self.target_transform:
            curr_output_img = self.target_transform(curr_output_img)

        input_imgs = torch.cat([prev_output_img, curr_input_img], dim=0)

        return input_imgs, curr_output_img
