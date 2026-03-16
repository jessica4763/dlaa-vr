import os
from pathlib import Path
import math
import numpy as np
import random
import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import OmegaConf

from datasets.qualcomm_dataset import QualcommDataset
from datasets.vr_dataset import VRDataset
from evaluate import run
from loss import CVVDPLoss, L1LossWithCVVDP
from models.qualcomm_network import QualcommNetwork
from models.vr_spatial_temporal_network import VRSpatialTemporalNetwork
from samplers.qualcomm_dataset_sampler import QualcommDatasetSampler
from utils import VRConfig, checkpoint, checkpoint_vr, gamma_to_linear, linear_to_gamma


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_epoch(
    device: str,
    model: nn.Module,
    training_dataloader: DataLoader,
    actual_batch_size: int,
    virtual_batch_size: int,
    clip_size: int,
    loss_function: nn.Module,
    padding: int,
    optimiser: MultiStepLR | CosineAnnealingLR,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    writer: SummaryWriter,
    iterations: int,
    use_jitter: bool
) -> None:
    total_instances = training_dataloader.dataset.total_instances

    total_loss = 0

    accumulation_steps = virtual_batch_size // actual_batch_size

    model.train()
    for batch, (inputs, motion_vectors, jitter, targets, curr_frame_num) in enumerate(training_dataloader):
        # input_N = num_batches * clip_size
        input_N, input_C, input_H, input_W = inputs.shape
        inputs = inputs.to(device, non_blocking=True)  # non_blocking=True requires pin_memory=True
        inputs = inputs.view(-1, clip_size, input_C, input_H, input_W)

        # output_N = num_batches * clip_size. output_H == input_H and output_W == input_W with no upscaling
        output_N, output_C, output_H, output_W = targets.shape
        motion_vectors = motion_vectors.to(device, non_blocking=True)
        motion_vectors = motion_vectors.view(-1, clip_size, 2, output_H, output_W)

        if use_jitter: 
            jitter = jitter.to(device, non_blocking=True)
            jitter = jitter.view(-1, clip_size, 2)
        else:
            jitter = None

        targets = targets.to(device, non_blocking=True)

        # Pass in motion vectors as well, for warping
        pred_frames, _, _ = model(inputs, motion_vectors, curr_frame_num, jitter, "training")
        pred_frames = pred_frames.view(-1, output_C, output_H, output_W)

        # Compute loss on only the centre of the patch
        pred_frames = pred_frames[:, :, padding:output_H - padding, padding:output_W - padding]
        targets = targets[:, :, padding:output_H - padding, padding:output_W - padding]
        gamma_pred_frames = linear_to_gamma(pred_frames)
        gamma_targets = linear_to_gamma(targets)
        loss = loss_function(gamma_pred_frames, gamma_targets) / accumulation_steps
        loss.backward()

        # For reporting
        total_loss += loss.item()

        if (batch + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if grad_norm > 1.0:
                print(f"{grad_norm=}")

            optimiser.step()
            optimiser.zero_grad()

            scheduler.step()

            writer.add_scalar(
                "loss/train",
                total_loss,
                iterations + (batch + 1) // accumulation_steps
            )
            print(f"Loss: {total_loss:>7f} [{min((batch + 1) * actual_batch_size, total_instances):>5d} / {total_instances:>5d}]")

            total_loss = 0


def train_epoch_vr(
    device: str,
    model: nn.Module,
    training_dataloader: DataLoader,
    actual_batch_size: int,
    virtual_batch_size: int,
    clip_size: int,
    loss_function: nn.Module,
    padding: int,
    optimiser: MultiStepLR | CosineAnnealingLR,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    writer: SummaryWriter,
    iterations: int,
    use_jitter: bool
) -> None:
    total_instances = training_dataloader.dataset.total_instances

    total_loss = 0

    accumulation_steps = virtual_batch_size // actual_batch_size

    model.train()
    for batch, data in enumerate(training_dataloader):
        # -------------------------------------------------------------------------
        # ----------------------------- Prepare inputs ----------------------------
        # -------------------------------------------------------------------------
        (
            left_inputs,
            right_inputs, 
            left_motion_vectors, 
            right_motion_vectors, 
            prev_left_depth,
            prev_right_depth,
            jitter, 
            left_targets, 
            right_targets, 
            curr_frame_num
        ) = data

        # input_N = num_batches * clip_size
        input_N, input_C, input_H, input_W = left_inputs.shape
        left_inputs = left_inputs.to(device, non_blocking=True)  # non_blocking=True requires pin_memory=True
        left_inputs = left_inputs.view(-1, clip_size, input_C, input_H, input_W)

        right_inputs = right_inputs.to(device, non_blocking=True)  # non_blocking=True requires pin_memory=True
        right_inputs = right_inputs.view(-1, clip_size, input_C, input_H, input_W)

        # input_N = num_batches * clip_size
        input_N, input_C, input_H, input_W = prev_left_depth.shape
        prev_left_depth = prev_left_depth.to(device, non_blocking=True)
        prev_left_depth = prev_left_depth.view(-1, clip_size, input_C, input_H, input_W)

        prev_right_depth = prev_right_depth.to(device, non_blocking=True)
        prev_right_depth = prev_right_depth.view(-1, clip_size, input_C, input_H, input_W)

        # output_N = num_batches * clip_size. output_H == input_H and output_W == input_W with no upscaling
        input_N, input_C, input_H, input_W = left_motion_vectors.shape
        left_motion_vectors = left_motion_vectors.to(device, non_blocking=True)
        left_motion_vectors = left_motion_vectors.view(-1, clip_size, input_C, input_H, input_W)

        right_motion_vectors = right_motion_vectors.to(device, non_blocking=True)
        right_motion_vectors = right_motion_vectors.view(-1, clip_size, input_C, input_H, input_W)

        if use_jitter: 
            jitter = jitter.to(device, non_blocking=True)
            jitter = jitter.view(-1, clip_size, 2)
        else:
            jitter = None

        left_targets = left_targets.to(device, non_blocking=True)
        right_targets = right_targets.to(device, non_blocking=True)

        # Pass in motion vectors as well, for warping
        pred_left_frames, pred_right_frames, _, _, _, _ = model(
            left_inputs,
            right_inputs,
            left_motion_vectors,
            right_motion_vectors,
            prev_left_depth,
            prev_right_depth,
            curr_frame_num,
            jitter,
            "training"
        )

        output_N, output_C, output_H, output_W = left_targets.shape
        pred_left_frames = pred_left_frames.view(-1, output_C, output_H, output_W)
        pred_right_frames = pred_right_frames.view(-1, output_C, output_H, output_W)

        # Compute loss on only the centre of the patch
        pred_left_frames = pred_left_frames[:, :, padding:output_H - padding, padding:output_W - padding]
        left_targets = left_targets[:, :, padding:output_H - padding, padding:output_W - padding]
        pred_right_frames = pred_right_frames[:, :, padding:output_H - padding, padding:output_W - padding]
        right_targets = right_targets[:, :, padding:output_H - padding, padding:output_W - padding]

        gamma_pred_left_frames = linear_to_gamma(pred_left_frames)
        gamma_left_targets = linear_to_gamma(left_targets)
        gamma_pred_right_frames = linear_to_gamma(pred_right_frames)
        gamma_right_targets = linear_to_gamma(right_targets)
        
        left_frame_loss = loss_function(gamma_pred_left_frames, gamma_left_targets) / accumulation_steps
        right_frame_loss = loss_function(gamma_pred_right_frames, gamma_right_targets) / accumulation_steps
        loss = (left_frame_loss + right_frame_loss) / 2
        loss.backward()

        # For reporting
        total_loss += loss.item()

        if (batch + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if grad_norm > 1.0:
                print(f"{grad_norm=}")

            optimiser.step()
            optimiser.zero_grad()

            scheduler.step()

            writer.add_scalar(
                "loss/train",
                total_loss,
                iterations + (batch + 1) // accumulation_steps
            )
            print(f"Loss: {total_loss:>7f} [{min((batch + 1) * actual_batch_size, total_instances):>5d} / {total_instances:>5d}]")

            total_loss = 0


def train() -> None:
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="train")

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    checkpoints_path = Path(cfg["paths"]["checkpoints-path"])
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    sanity_checks_output_path = Path(cfg["paths"]["sanity-checks-output-path"])
    sanity_checks_output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # ---------------------------- Reproducibility ----------------------------
    # -------------------------------------------------------------------------
    random.seed(cfg["setup"]["seed"])
    np.random.seed(cfg["setup"]["seed"])

    # Seeds both the CPU and CUDA
    torch.manual_seed(cfg["setup"]["seed"])

    torch.use_deterministic_algorithms(True)

    # Deterministically selecting an algorithm reduces efficiency
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = True

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # -------------------------------------------------------------------------
    # ------------------- Training + validation diagnostics -------------------
    # -------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg["paths"]["tensorboard-path"])
    
    # Log the config
    writer.add_text("hyperparams", OmegaConf.to_yaml(cfg))

    # -------------------------------------------------------------------------
    # ------------------------------- Constants -------------------------------
    # -------------------------------------------------------------------------
    scale_factor = cfg["dataset"]["output-frame-height"] // cfg["dataset"]["input-frame-height"]

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    if cfg["dataset"]["is-vr"]:
        training_data = VRDataset(
            input_imgs_path=cfg["dataset"]["training-input-img-path"],
            output_imgs_path=cfg["dataset"]["training-output-img-path"],
            input_frame_height=cfg["dataset"]["input-frame-height"],
            input_frame_width=cfg["dataset"]["input-frame-width"],
            camera_data_path_suffix=cfg["dataset"]["camera-data-path-suffix"],
            input_path_suffix=cfg["dataset"]["input-path-suffix"],
            jittered_input_path_suffix=cfg["dataset"]["jittered-input-path-suffix"],
            colour_path_suffix=cfg["dataset"]["colour-path-suffix"],
            depth_path_suffix=cfg["dataset"]["depth-path-suffix"],
            motion_vector_path_suffix=cfg["dataset"]["motion-vector-path-suffix"],
            scene_names=cfg["dataset"]["scene-names"],
            use_jitter=cfg["setup"]["jitter"],
            scale_factor=scale_factor,
            dilation_block_size=cfg["dataset"]["dilation-block-size"],
            transform=gamma_to_linear,
            target_transform=gamma_to_linear,
            mode="training"
        )
    else:
        training_data = QualcommDataset(
            input_imgs_path=cfg["dataset"]["training-input-img-path"],
            output_imgs_path=cfg["dataset"]["training-output-img-path"],
            input_frame_height=cfg["dataset"]["input-frame-height"],
            input_frame_width=cfg["dataset"]["input-frame-width"],
            camera_data_path_suffix=cfg["dataset"]["camera-data-path-suffix"],
            colour_path_suffix=cfg["dataset"]["colour-path-suffix"],
            depth_path_suffix=cfg["dataset"]["depth-path-suffix"],
            motion_vector_path_suffix=cfg["dataset"]["motion-vector-path-suffix"],
            colour_jittered_path_suffix=cfg["dataset"]["colour-jittered-path-suffix"],
            depth_jittered_path_suffix=cfg["dataset"]["depth-jittered-path-suffix"],
            motion_vector_jittered_path_suffix=cfg["dataset"]["motion-vector-jittered-path-suffix"],
            scene_names=cfg["dataset"]["scene-names"],
            use_jitter=cfg["setup"]["jitter"],
            scale_factor=scale_factor,
            dilation_block_size=cfg["dataset"]["dilation-block-size"],
            transform=gamma_to_linear,
            target_transform=gamma_to_linear,
            mode="training"
        )

    training_sampler = QualcommDatasetSampler(
        scenes=training_data.scenes,
        instance_boundaries=training_data.instance_boundaries,
        total_instances=training_data.total_instances,
        frame_boundaries=training_data.frame_boundaries,
        total_frames=training_data.total_frames,
        batch_size=cfg["optimiser"]["actual-batch-size"],
        clip_size=cfg["optimiser"]["clip-size"],
        input_frame_height=cfg["dataset"]["input-frame-height"],
        input_frame_width=cfg["dataset"]["input-frame-width"],
        high_res_patch_size=cfg["optimiser"]["patch-size"],
        padding=cfg["optimiser"]["padding"],
        scale_factor=scale_factor
    )
    
    # For reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    training_dataloader = DataLoader(
        training_data,
        batch_sampler=training_sampler,
        num_workers=os.cpu_count() - 4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    iterations_per_epoch = math.ceil(training_data.total_instances / cfg["optimiser"]["virtual-batch-size"])

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    if cfg["dataset"]["is-vr"]:
        vr_config = VRConfig(
            camera_baseline=cfg["dataset"]["camera-baseline"], 
            horizontal_fov=cfg["dataset"]["horizontal-fov"],
            vertical_fov=cfg["dataset"]["vertical-fov"],
            horizontal_resolution=cfg["dataset"]["input-frame-width"],
            vertical_resolution=cfg["dataset"]["input-frame-height"]
        )

        model = VRSpatialTemporalNetwork(
            vr_config=vr_config,
            hidden_channels=cfg["model"]["hidden-channels"],
            num_blocks=cfg["model"]["num-blocks"],
            scale_factor=scale_factor,
            use_jitter=cfg["setup"]["jitter"]
        ).to(device)
    else:
        vr_config = None

        model = QualcommNetwork(
            hidden_channels=cfg["model"]["hidden-channels"],
            num_blocks=cfg["model"]["num-blocks"],
            scale_factor=scale_factor,
            use_jitter=cfg["setup"]["jitter"]
        ).to(device)

    # Initialise with parameters from a previously trained model if desired
    parameters_path = cfg["model"]["parameters"]
    if parameters_path:
        model.load_state_dict(
            torch.load(
                parameters_path,
                weights_only=True,
                map_location=device
            )
        )

    # -------------------------------------------------------------------------
    # ----------------------------- Optimisation ------------------------------
    # -------------------------------------------------------------------------
    if cfg["optimiser"]["loss"] == "l1loss":
        loss_function = nn.L1Loss()
    elif cfg["optimiser"]["loss"] == "mseloss":
        loss_function = nn.MSELoss()
    elif cfg["optimiser"]["loss"] == "cvvdploss":
        loss_function = CVVDPLoss(cfg["setup"]["display-name"])
    elif cfg["optimiser"]["loss"] == "l1loss_with_cvvdp":
        loss_function = L1LossWithCVVDP(
            cfg["setup"]["display-name"],
            cvvdp_weight=cfg["optimiser"]["cvvdp-weight"]
        )
    else:
        sys.exit("Chosen loss function does not exist.")

    if cfg["optimiser"]["name"] == "adamw":
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimiser"]["learning-rate"],
            betas=cfg["optimiser"]["betas"],
            eps=cfg["optimiser"]["epsilon"],
            weight_decay=cfg["optimiser"]["weight-decay"]
        )
    else:
        sys.exit("Chosen optimiser implementation does not exist.")

    if cfg["optimiser"]["lr-scheduler"] == "multi-step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimiser,
            milestones=cfg["optimiser"]["learning-rate-milestones"],
            gamma=cfg["optimiser"]["learning-rate-gamma"]
        )
    elif cfg["optimiser"]["lr-scheduler"] == "cosine-annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            cfg["optimiser"]["iterations"],
            cfg["optimiser"]["learning-rate-eta-min"]
        )
    else:
        sys.exit("Chosen learning rate scheduler implementation does not exist.")

    # -------------------------------------------------------------------------
    # -------------------- Load from a training checkpoint --------------------
    # -------------------------------------------------------------------------
    # If the training checkpoint exists
    if os.path.exists(cfg["paths"]["training-checkpoint-path"]):
        training_checkpoint = torch.load(cfg["paths"]["training-checkpoint-path"], weights_only=False)

        epoch = training_checkpoint["epoch"] + 1
        iterations = epoch * iterations_per_epoch + 1
        model.load_state_dict(training_checkpoint["model"])
        optimiser.load_state_dict(training_checkpoint["optimiser"])
        scheduler.load_state_dict(training_checkpoint["scheduler"])

        torch.set_rng_state(training_checkpoint["rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(training_checkpoint["cuda_rng_state"])

        print(f"Resuming from epoch {epoch} / iteration {iterations} at learning rate: {scheduler.get_last_lr()[0]}")
    else:
        epoch = iterations = 0
        print("No training checkpoint.")

    # model = torch.compile(model)

    # -------------------------------------------------------------------------
    # ---------------------- Training + validation loop -----------------------
    # -------------------------------------------------------------------------
    # torch.autograd.set_detect_anomaly(True)

    while iterations < cfg["optimiser"]["iterations"]:
        iterations = epoch * iterations_per_epoch + 1

        # -------------------------------------------------------------------------
        # ------------------------------- Training --------------------------------
        # -------------------------------------------------------------------------
        print(f"Epoch {epoch + 1} | Iteration {iterations} \n-------------------------------")

        if cfg["dataset"]["is-vr"]:
            train_epoch_vr(
                device=device,
                model=model,
                training_dataloader=training_dataloader,
                actual_batch_size=cfg["optimiser"]["actual-batch-size"],
                virtual_batch_size=cfg["optimiser"]["virtual-batch-size"],
                clip_size=cfg["optimiser"]["clip-size"],
                loss_function=loss_function,
                padding=cfg["optimiser"]["padding"],
                optimiser=optimiser,
                scheduler=scheduler,
                writer=writer,
                iterations=iterations,
                use_jitter=cfg["setup"]["jitter"]
            )
        else:
            train_epoch(
                device=device,
                model=model,
                training_dataloader=training_dataloader,
                actual_batch_size=cfg["optimiser"]["actual-batch-size"],
                virtual_batch_size=cfg["optimiser"]["virtual-batch-size"],
                clip_size=cfg["optimiser"]["clip-size"],
                loss_function=loss_function,
                padding=cfg["optimiser"]["padding"],
                optimiser=optimiser,
                scheduler=scheduler,
                writer=writer,
                iterations=iterations,
                use_jitter=cfg["setup"]["jitter"]
            )

        if cfg["dataset"]["is-vr"]:
            checkpoint_vr(
                checkpoints_path=checkpoints_path,
                sanity_checks_output_path=sanity_checks_output_path,
                device=device,
                model=model,
                data=training_data,
                vr_config=vr_config,
                iterations=iterations,
                iterations_per_epoch=iterations_per_epoch,
                input_frame_height=cfg["dataset"]["input-frame-height"],
                input_frame_width=cfg["dataset"]["input-frame-width"],
                scale_factor=scale_factor,
                use_jitter=cfg["setup"]["jitter"],
                mode="training"
            )
        else:
            checkpoint(
                checkpoints_path=checkpoints_path,
                sanity_checks_output_path=sanity_checks_output_path,
                device=device,
                model=model,
                data=training_data,
                iterations=iterations,
                iterations_per_epoch=iterations_per_epoch,
                input_frame_height=cfg["dataset"]["input-frame-height"],
                input_frame_width=cfg["dataset"]["input-frame-width"],
                scale_factor=scale_factor,
                use_jitter=cfg["setup"]["jitter"],
                mode="training"
            )

        # -------------------------------------------------------------------------
        # ------------------------- Training checkpoint ---------------------------
        # -------------------------------------------------------------------------
        model_temp_file = cfg["paths"]["saved-models-path"] + ".tmp"
        torch.save(model.state_dict(), model_temp_file)
        with open(model_temp_file, 'ab') as f:
            os.fsync(f.fileno())
        os.replace(model_temp_file, cfg["paths"]["saved-models-path"])

        training_checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        training_checkpoint_temp_file = cfg["paths"]["training-checkpoint-path"] + ".tmp"
        torch.save(training_checkpoint, training_checkpoint_temp_file)
        with open(training_checkpoint_temp_file, 'ab') as f:
            os.fsync(f.fileno())
        os.replace(training_checkpoint_temp_file, cfg["paths"]["training-checkpoint-path"])

        save_interval = 10000
        if iterations % save_interval < iterations_per_epoch:
            id = (iterations // save_interval) * save_interval

            model_save_path = checkpoints_path / f"model_{id}.pt"
            torch.save(model.state_dict(), model_save_path)
            with open(model_save_path, 'ab') as f:
                os.fsync(f.fileno())

            training_checkpoint_save_path = checkpoints_path / f"training_checkpoint_{id}.pt"
            torch.save(training_checkpoint, training_checkpoint_save_path)
            with open(training_checkpoint_save_path, 'ab') as f:
                os.fsync(f.fileno())

        # -------------------------------------------------------------------------
        # ------------------------------ Validation -------------------------------
        # -------------------------------------------------------------------------

        # Proxy validation happens every 1000 iterations, whereas primary validation happens every 100000 iterations
        if cfg["dataset"]["is-vr"]:
            validate_vr(
                iterations=iterations + iterations_per_epoch,
                iterations_per_epoch=iterations_per_epoch,
                primary_validation_interval=cfg["validation"]["primary-validation-interval"],
                proxy_validation_interval=cfg["validation"]["proxy-validation-interval"],
                writer=writer,
                saved_models_path=cfg["paths"]["saved-models-path"]
            )
        else:
            validate(
                iterations=iterations + iterations_per_epoch,
                iterations_per_epoch=iterations_per_epoch,
                primary_validation_interval=cfg["validation"]["primary-validation-interval"],
                proxy_validation_interval=cfg["validation"]["proxy-validation-interval"],
                writer=writer,
                saved_models_path=cfg["paths"]["saved-models-path"]
            )

        # -------------------------------------------------------------------------
        # ------------------------- Update training state -------------------------
        # -------------------------------------------------------------------------
        epoch += 1

    # -------------------------------------------------------------------------
    # -------------------------------- Cleanup --------------------------------
    # ------------------------------------------------------------------------- 
    writer.flush()
    writer.close()


def validate_vr(
    iterations: int,
    iterations_per_epoch: int,
    primary_validation_interval: int,
    proxy_validation_interval: int,
    writer: SummaryWriter,
    saved_models_path: str
) -> None:
    with hydra.initialize(version_base=None, config_path="../configs"):
        if iterations % primary_validation_interval < iterations_per_epoch:
            validation_cfg = hydra.compose(
                config_name="vr-validation", 
                overrides=[
                    "dataset=vr-validation-upscale-fantasticvillage",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)
        elif iterations % proxy_validation_interval < iterations_per_epoch:
            validation_cfg = hydra.compose(
                config_name="vr-validation", 
                overrides=[
                    "dataset=vr-validation-upscale-fantasticvillage",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)


def validate(
    iterations: int,
    iterations_per_epoch: int,
    primary_validation_interval: int,
    proxy_validation_interval: int,
    writer: SummaryWriter,
    saved_models_path: str
) -> None:
    with hydra.initialize(version_base=None, config_path="../configs"):
        if iterations % primary_validation_interval < iterations_per_epoch:
            validation_cfg = hydra.compose(
                config_name="validation", 
                overrides=[
                    "dataset=validation-upscale-seaport",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)

            validation_cfg = hydra.compose(
                config_name="validation", 
                overrides=[
                    "dataset=validation-upscale-abandonedschool",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)

            validation_cfg = hydra.compose(
                config_name="validation", 
                overrides=[
                    "dataset=validation-upscale-spaceshipdemo",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)

            validation_cfg = hydra.compose(
                config_name="validation", 
                overrides=[
                    "dataset=validation-upscale-stationary-segment",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)
        elif iterations % proxy_validation_interval < iterations_per_epoch:
            validation_cfg = hydra.compose(
                config_name="validation", 
                overrides=[
                    "dataset=validation-upscale-seaport",
                    f"paths.saved-models-path={saved_models_path}"
                ]
            )
            run(cfg=validation_cfg, writer=writer, iterations=iterations)


if __name__ == "__main__":
    train()
