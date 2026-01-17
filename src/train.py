import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from datasets import QualcommDataset
from loss import CVVDPLoss, L1LossWithCVVDP
from model import QualcommNetwork
from samplers import QualcommDatasetSampler
from sanity_checks import save_input, save_output
from utils import gamma_to_linear, linear_to_gamma


def train_epoch(
    device: str,
    model: nn.Module,
    training_dataloader: DataLoader,
    actual_batch_size: int,
    virtual_batch_size: int,
    clip_size: int,
    loss_function: nn.Module,
    optimiser: optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    use_jitter: bool = False
) -> None:
    total_instances = training_dataloader.dataset.total_instances

    total_loss = 0

    accumulation_steps = virtual_batch_size // actual_batch_size

    model.train()
    for batch, (inputs, motion_vectors, jitter, targets) in enumerate(training_dataloader):
        # input_N = num_batches * clip_size
        input_N, input_C, input_H, input_W = inputs.shape
        inputs = inputs.to(device, non_blocking=True)
        inputs = inputs.view(-1, clip_size, input_C, input_H, input_W)

        # output_N = num_batches * clip_size. output_H == input_H and output_W == input_W for no upscaling
        output_N, output_C, output_H, output_W = motion_vectors.shape
        motion_vectors = motion_vectors.to(device, non_blocking=True)
        motion_vectors = motion_vectors.view(-1, clip_size, output_C, output_H, output_W)

        if use_jitter: 
            jitter = jitter.to(device, non_blocking=True)
            jitter = jitter.view(-1, clip_size, 2)
        else:
            jitter = None

        targets = targets.to(device, non_blocking=True)

        # Pass in motion vectors as well, for warping
        pred_frame, _ = model(inputs, motion_vectors, jitter)
        pred_frame = pred_frame.view(-1, 3, output_H, output_W)
        loss = loss_function(pred_frame, targets) / accumulation_steps
        loss.backward()

        total_loss += loss.item()

        if (batch + 1) % accumulation_steps == 0:
            optimiser.step()
            optimiser.zero_grad()

            writer.add_scalar(
                "loss/train",
                total_loss,
                epoch * total_instances + batch
            )

            print(f"Loss: {total_loss:>7f}  [{batch + 1:>5d} / {total_instances:>5d}]")

            total_loss = 0


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
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
    torch.manual_seed(cfg["setup"]["seed"])

    # Deterministically selecting an algorithm reduces efficiency
    torch.backends.cudnn.benchmark = True

    torch.use_deterministic_algorithms(False)

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    # -------------------------------------------------------------------------
    # ------------------------------ Diagnostics ------------------------------
    # -------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg["paths"]["tensorboard-path"])
    
    # Log the config
    writer.add_text("hyperparams", OmegaConf.to_yaml(cfg))

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    training_data = QualcommDataset(
        input_imgs_path=cfg["dataset"]["training-input-img-path"],
        output_imgs_path=cfg["dataset"]["training-output-img-path"],
        input_frame_height=cfg["dataset"]["input-frame-height"],
        input_frame_width=cfg["dataset"]["input-frame-width"],
        output_frame_height=cfg["dataset"]["output-frame-height"],
        output_frame_width=cfg["dataset"]["output-frame-width"],
        camera_data_path_suffix=cfg["dataset"]["camera-data-path-suffix"],
        ground_truth_path_suffix=cfg["dataset"]["ground-truth-path-suffix"],
        colour_path_suffix=cfg["dataset"]["colour-path-suffix"],
        depth_path_suffix=cfg["dataset"]["depth-path-suffix"],
        motion_vector_path_suffix=cfg["dataset"]["motion-vector-path-suffix"],
        colour_jittered_path_suffix=cfg["dataset"]["colour-jittered-path-suffix"],
        depth_jittered_path_suffix=cfg["dataset"]["depth-jittered-path-suffix"],
        motion_vector_jittered_path_suffix=cfg["dataset"]["motion-vector-jittered-path-suffix"],
        scene_names=cfg["dataset"]["scene_names"],
        use_jitter=cfg["setup"]["jitter"],
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
        output_frame_height=cfg["dataset"]["output-frame-height"],
        output_frame_width=cfg["dataset"]["output-frame-width"],
        high_res_patch_size=cfg["optimiser"]["patch-size"]
    )

    training_dataloader = DataLoader(
        training_data,
        batch_sampler=training_sampler,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg["model"]["hidden-channels"],
        num_blocks=cfg["model"]["num-blocks"],
        input_frame_height=cfg["dataset"]["input-frame-height"],
        input_frame_width=cfg["dataset"]["input-frame-width"],
        output_frame_height=cfg["dataset"]["output-frame-height"],
        output_frame_width=cfg["dataset"]["output-frame-width"],
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

    # model = torch.compile(model)

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

    if cfg["optimiser"]["name"] == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=cfg["optimiser"]["learning-rate"],
            weight_decay=cfg["optimiser"]["weight-decay"]
        )
    elif cfg["optimiser"]["name"] == "adamw":
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
            cfg["optimiser"]["epochs"],
            cfg["optimiser"]["learning-rate-eta-min"]
        )

    # -------------------------------------------------------------------------
    # ----------------------------- Training loop -----------------------------
    # -------------------------------------------------------------------------
    for epoch in range(cfg["optimiser"]["epochs"]):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_epoch(
            device=device,
            model=model,
            training_dataloader=training_dataloader,
            actual_batch_size=cfg["optimiser"]["actual-batch-size"],
            virtual_batch_size=cfg["optimiser"]["virtual-batch-size"],
            clip_size=cfg["optimiser"]["clip-size"],
            loss_function=loss_function,
            optimiser=optimiser,
            writer=writer,
            epoch=epoch,
            use_jitter=cfg["setup"]["jitter"]
        )

        checkpoint(
            checkpoint_path=checkpoints_path,
            sanity_checks_output_path=sanity_checks_output_path,
            device=device,
            model=model,
            training_data=training_data,
            writer=writer,
            epoch=epoch,
            input_frame_height=cfg["dataset"]["input-frame-height"],
            input_frame_width=cfg["dataset"]["input-frame-width"],
            output_frame_height=cfg["dataset"]["output-frame-height"],
            output_frame_width=cfg["dataset"]["output-frame-width"],
            use_jitter=cfg["setup"]["jitter"]
        )

        scheduler.step()

        # Save the model after each epoch
        torch.save(model.state_dict(), Path(cfg["paths"]["saved-models-path"]))

    writer.flush()
    writer.close()

    print("Done.")


def checkpoint(
    checkpoint_path: Path,
    sanity_checks_output_path: Path,
    device: str,
    model: nn.Module,
    training_data: Dataset,
    writer: SummaryWriter,
    epoch: int,
    input_frame_height: int,
    input_frame_width: int,
    output_frame_height: int,
    output_frame_width: int,
    use_jitter: bool = False
) -> None:
    # Strictly a training diagnostic, so it's OK if
    # training data is used here
    model.eval()
    with torch.no_grad():
        inputs, motion_vectors, jitter, output = training_data[(0, 0, 0, input_frame_width, input_frame_height)]

        # Verify input to the network
        scale_factor = output_frame_height // input_frame_height
        save_input(sanity_checks_output_path, model, inputs, motion_vectors, scale_factor)

        # Verify the goal of the network
        output = linear_to_gamma(output)
        save_output(sanity_checks_output_path, output)

        inputs = inputs.to(device).unsqueeze(0).unsqueeze(0)
        motion_vectors = motion_vectors.to(device).unsqueeze(0).unsqueeze(0)
        jitter = jitter.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None
        anti_aliased_img, _ = model(inputs, motion_vectors, jitter)
        anti_aliased_img = anti_aliased_img.squeeze(0).squeeze(0)
        anti_aliased_img = linear_to_gamma(anti_aliased_img)

        save_image(anti_aliased_img, checkpoint_path / f"{epoch}.png")
        if (epoch + 1) % 10 == 0:
            writer.add_image(
                "checkpoint images",
                anti_aliased_img,
                global_step=(epoch + 1)
            )


if __name__ == "__main__":
    train()
