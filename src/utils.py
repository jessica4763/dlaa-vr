import torch


def gamma_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32) / 255.0

    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )


def linear_to_gamma(image: torch.Tensor) -> torch.Tensor:
    image = torch.clamp(image, 0.0, 1.0)
    return torch.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * (image ** (1.0 / 2.4)) - 0.055
    )


def cumsum(xs):
    cumsum_xs = [0]
    for x in xs:
        cumsum_xs.append(cumsum_xs[-1] + x)

    return cumsum_xs
