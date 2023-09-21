from experiments import generation
import matplotlib.pyplot as plt

# Set the weights path
weights_path = "convex_diffusion_celeba_grayscale.hdf5"

# Choose number of images to generate
n_images = 3

# Generate
mode = 'convex_unet'
x_gen = generation.generate_from_model(weights_path, n_images, mode)

print(x_gen.numpy().min(), x_gen.numpy().max())

# Save in png
plt.figure(figsize=(6*n_images, 6))
for i in range(n_images):
    plt.subplot(1, n_images, i+1)
    plt.imshow(x_gen[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.savefig(f'./results/{mode}_generation.png')
plt.close()