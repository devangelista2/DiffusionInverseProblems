from miscellaneous import configurations, utilities

# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo128.yml"
GENERATIVE_MODEL = "DCGAN"

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Define generative model
weights_path = f"./model_weights/{GENERATIVE_MODEL}/{config.data.dataset}_{config.training.loss}.pth"
model = utilities.get_model(GENERATIVE_MODEL, config, weights_path=weights_path)

# Test the model generation
model.test_generation(
    path=f"./results/{config.data.dataset}_{config.training.loss}",
    n_samples=16,
)
