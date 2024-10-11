import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])



def nn_params(model):
    """Get the number of parameters of a model

    References:
        https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_accuracy_loss(train_accuracies, val_accuracies, train_losses, val_losses):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_images(images, labels, preds, n):

    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            index = i * n + j
            # Normalize the image data
            image = images[index].permute(1, 2, 0)
            image = (image - image.min()) / (image.max() - image.min())
            ax[i, j].imshow(image)
            pred = preds[i].argmax().item()
            ax[i, j].set_title(f"{labels[index].item()} -> {pred}")
            ax[i, j].axis('off')
    plt.show()


def save_clean_model(model, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = f"{save_dir}/clean_model.pth"

    torch.save(model.state_dict(), model_path)

