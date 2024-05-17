import torch

from miscellaneous import configurations, data
from models.GAN import DCGAN

import time


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo128.yml"
TRAIN = True

SAVE_RESULTS = True

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load the data
x_train, _ = data.load_data(config)
print(x_train[0].shape)

# Set device
device = config.device

# Define GAN model
G = DCGAN.Generator(
    input_shape=(config.data.channels, config.data.image_size, config.data.image_size)
).to(device)
D = DCGAN.Discriminator(
    input_shape=(config.data.channels, config.data.image_size, config.data.image_size)
).to(device)

# Trainloader
train_loader = torch.utils.data.DataLoader(
    x_train, batch_size=config.training.batch_size, shuffle=True
)

# Define the optimizers
optimizerG = torch.optim.Adam(
    G.parameters(),
    lr=config.optimizer.lr,
    weight_decay=config.optimizer.weight_decay,
    betas=(config.optimizer.beta1, 0.999),
    eps=config.optimizer.eps,
)
optimizerD = torch.optim.Adam(
    D.parameters(),
    lr=config.optimizer.lr,
    weight_decay=config.optimizer.weight_decay,
    betas=(config.optimizer.beta1, 0.999),
    eps=config.optimizer.eps,
)

# Define the loss function
D_loss = torch.nn.BCELoss()
G_loss = torch.nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Compute the total number of steps per epoch
steps_per_epoch = len(x_train) // config.training.batch_size


# Train if required, else load weights
if TRAIN:
    # Start training
    for epoch in range(config.training.n_epochs):
        total_DLoss = 0
        total_GLoss = 0
        step = 0
        start_time = time.time()
        print(f"Epoch: {epoch+1}/{config.training.n_epochs}:")

        # Batch steps
        for i, x in enumerate(train_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            D.zero_grad()

            # Format batch
            x = x.to(device)

            # Compute batch size
            b_size = x.size(0)

            # Define label
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = D(x).view(-1)

            # Calculate loss on all-real batch
            errD_real = D_loss(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn_like(x, device=device)

            # Generate fake image batch with G
            fake = G(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = D(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = D_loss(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = D(fake).view(-1)

            # Calculate G's loss based on this output
            errG = D_loss(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            optimizerG.step()

            # Update step
            step = step + 1

            # Print out result
            total_DLoss = total_DLoss + D_G_z1
            total_GLoss = total_GLoss + D_G_z2
            time_elapsed = format_seconds(seconds=time.time() - start_time)
            print(
                f"Iteration: {step}/{steps_per_epoch}, Time elapsed: {time_elapsed}, DLoss: {total_DLoss/(i+1):0.4f}, GLoss: {total_GLoss/(i+1):0.4f}.",
                end="\r",
            )

        # Every 10 epochs, save the model weights
        if (epoch % 10) == 0:
            weights_path = (
                f"./model_weights/GAN/{config.data.dataset}_{config.training.loss}.pth"
            )
            torch.save(G.state_dict(), weights_path)
else:
    weights_path = (
        f"./model_weights/GAN/{config.data.dataset}_{config.training.loss}.pth"
    )
    G.model.load_state_dict(torch.load(weights_path))
