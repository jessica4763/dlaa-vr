from bisect import bisect_right
import random
from typing import Iterator
from torch.utils.data import Sampler


from utils import Scene


class QualcommDatasetSampler(Sampler[list[int]]):
    def __init__(
        self,
        scenes: list[Scene],
        instance_boundaries: list[int],
        total_instances: int,
        frame_boundaries: list[int],
        total_frames: int,
        batch_size: int,
        clip_size: int
    ) -> None:
        self.scenes = scenes
        self.instance_boundaries = instance_boundaries
        self.total_instances = total_instances
        self.frame_boundaries = frame_boundaries
        self.total_frames = total_frames
        self.batch_size = batch_size
        self.clip_size = clip_size

    def __len__(self) -> int:
        return self.total_instances

    def __iter__(self) -> Iterator[list[int]]:
        # Number and shuffle all of the instances
        instance_indices = list(range(self.total_instances))
        random.shuffle(instance_indices)

        for batch in range(0, self.total_instances, self.batch_size):
            # A batch consists of batch_size (or fewer) instances
            instances = instance_indices[batch:batch + self.batch_size]

            # For each of the chosen instances, randomly pick a continuous
            # section of frames to train on for this epoch, and store their
            # indices in frame_indices
            frame_indices = []
            for instance_idx in instances:
                # Get the scene associated with instance_idx
                scene_idx = bisect_right(self.instance_boundaries, instance_idx) - 1
                scene = self.scenes[scene_idx]

                # Offset the index relative to the scene
                idx_offset = self.instance_boundaries[scene_idx]
                relative_instance_idx = instance_idx - idx_offset

                instance_start = self.frame_boundaries[scene_idx] + relative_instance_idx * scene.num_frames_per_instance
                clip_start = random.randint(instance_start, instance_start + scene.num_frames_per_instance - self.clip_size)
                for frame_idx in range(clip_start, clip_start + self.clip_size):
                    frame_indices.append(frame_idx)

            yield frame_indices
