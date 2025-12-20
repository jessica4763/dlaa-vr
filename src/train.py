import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from scipy import stats
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from datasets import QualcommDataset, QualcommDatasetSampler
from model import QualcommNetwork
from sanity_checks import output_input
from utils import gamma_to_linear, linear_to_gamma


def apply_learning_curriculum(
    learning_curriculum: str, 
    model: nn.Module,
    X: torch.Tensor,
    prev_frame_num: int,
    prev_pred_frame: torch.Tensor,
    prev_features: torch.Tensor,
    total_batches: int,
    epsilon: float,
    k: float,
    c: float
) -> torch.Tensor:
    if prev_pred_frame is None or prev_features is None:
        return X

    if learning_curriculum == "teacher-forcing":
        return X 
    elif learning_curriculum == "scheduled-sampling":
        p = max(epsilon, k - c * total_batches)
        x = stats.uniform.rvs(loc=0, scale=1, size=1)
        if x < p:
            return X
            
    # Use the predicted frame
    c0 = model.in_channels - (model.num_prev_colour + model.num_prev_feature)
    c1 = model.in_channels - model.num_prev_feature
    mask = prev_frame_num != 0
    X[mask, c0:c1] = prev_pred_frame[mask, ...].detach()
    X[mask, c1:model.in_channels] = prev_features[mask, ...].detach()

    return X


def train_epoch(
    device: str,
    model: nn.Module,
    training_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    learning_curriculum: str,
    scheduled_sampling_epsilon: float,
    scheduled_sampling_k: float,
    scheduled_sampling_c: float,
    writer: SummaryWriter,
    epoch: int
) -> None:
    dataset_size = len(training_dataloader.dataset)

    prev_pred_frame = prev_features = None

    model.train()
    for batch, (X, y, prev_frame_num) in enumerate(training_dataloader):
        X, y = X.to(device), y.to(device)

        total_batches = epoch * dataset_size + batch

        X = apply_learning_curriculum(
            learning_curriculum,
            model,
            X,
            prev_frame_num,
            prev_pred_frame,
            prev_features,
            total_batches,
            scheduled_sampling_epsilon,
            scheduled_sampling_k,
            scheduled_sampling_c
        )
            
        pred_frame, features = model(X)
        loss = loss_fn(pred_frame, y)

        prev_pred_frame = pred_frame
        prev_features = features

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar(
            "loss/train",
            loss.item(),
            total_batches
        )

        loss, current_img = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}  [{current_img:>5d}/{dataset_size:>5d}]")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # ---------------------------- Reproducibility ----------------------------
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg["setup"]["seed"])

    # Deterministcally selects an algorithm; reduces efficiency
    torch.backends.cudnn.benchmark = False

    # Use only deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    # -------------------------------------------------------------------------
    # ------------------------------ Diagnostics ------------------------------
    # -------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg["setup"]["tensorboard-dir"])

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    training_data = QualcommDataset(
        cfg["dataset"]["scene_names"],
        cfg["dataset"]["training-input-img-path"],
        cfg["dataset"]["training-output-img-path"],
        transform=gamma_to_linear,
        target_transform=gamma_to_linear,
    )

    training_sampler = QualcommDatasetSampler(
        training_data,
        cfg["optimiser"]["batch-size"],
    )

    training_dataloader = DataLoader(
        training_data,
        batch_sampler=training_sampler
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg["model"]["hidden-channels"],
        num_blocks=cfg["model"]["num-blocks"]
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
    loss_function = nn.L1Loss()

    if cfg["optimiser"]["name"] == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=cfg["optimiser"]["learning-rate"],
            weight_decay=cfg["optimiser"]["regularisation-parameter"]
        )
    elif cfg["optimiser"]["name"] == "adamw":
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimiser"]["learning-rate"],
            betas=cfg["optimiser"]["betas"],
            eps=cfg["optimiser"]["epsilon"],
            weight_decay=cfg["optimiser"]["regularisation-parameter"]
        )
    else:
        sys.exit("Chosen optimiser implementation does not exist.")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, 
        milestones=cfg["optimiser"]["learning-rate-milestones"],
        gamma=cfg["optimiser"]["learning-rate-gamma"]
    )

    # -------------------------------------------------------------------------
    # ----------------------------- Training loop -----------------------------
    # -------------------------------------------------------------------------
    for epoch in range(cfg["optimiser"]["epochs"]):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_epoch(
            device,
            model,
            training_dataloader,
            loss_function,
            optimiser,
            cfg["optimiser"]["learning-curriculum"],
            cfg["optimiser"]["scheduled-sampling-epsilon"],
            cfg["optimiser"]["scheduled-sampling-k"],
            cfg["optimiser"]["scheduled-sampling-c"],
            writer,
            epoch
        )
        checkpoint(
            device,
            model,
            training_data,
            writer,
            epoch
        )
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), Path(cfg["setup"]["saved-models-path"]))

    # Log the config
    writer.add_text("hyperparams", OmegaConf.to_yaml(cfg))

    writer.flush()
    writer.close()

    print("Done.")


def checkpoint(
    device: str,
    model: nn.Module,
    training_data: Dataset,
    writer: SummaryWriter,
    epoch: int
) -> None:
    # Strictly a training diagnostic, so it's OK if
    # training data is used here

    model.eval()
    with torch.no_grad():
        input_imgs, _, _= training_data[0]

        # Verify what exactly goes into the network
        if epoch == 0:
            output_input(model, input_imgs)

        input_imgs = input_imgs.to(device).unsqueeze(0)
        anti_aliased_img, _ = model(input_imgs)
        anti_aliased_img = anti_aliased_img.squeeze(0)
        anti_aliased_img = linear_to_gamma(anti_aliased_img)

        save_image(anti_aliased_img, f"checkpoints/{epoch}.png")
        if (epoch + 1) % 10 == 0:
            writer.add_image(
                "checkpoint images",
                anti_aliased_img,
                global_step=(epoch + 1)
            )


if __name__ == "__main__":
    train()
