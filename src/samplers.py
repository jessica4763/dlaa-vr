from bisect import bisect_right
import random
from typing import Iterator
from torch.utils.data import Sampler


from utils import Scene, cumsum


class QualcommDatasetSampler(Sampler[list[int]]):
    def __init__(
        self,
        scenes: list[Scene],
        batch_size: int,
        clip_size: int
    ):
        self.scenes = scenes
        self.total_scenes = len(scenes)

        scene_num_instances = [scene.num_instances for scene in self.scenes]
        self.instance_boundaries = cumsum(scene_num_instances)
        self.total_instances = sum(scene_num_instances)

        scene_num_frames = [scene.num_frames for scene in self.scenes]
        self.frame_boundaries = cumsum(scene_num_frames)
        self.total_frames = sum(scene_num_frames)

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
