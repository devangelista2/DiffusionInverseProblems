from miscellaneous import configurations, data, utilities

# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo256.yml"
GENERATIVE_MODEL = "ImprovedDDIM"
TRAIN = True

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load the data
x_train, _ = data.load_data(config)

# Train if required, else load weigths
if TRAIN:
    # Define generative model
    model = utilities.get_model(GENERATIVE_MODEL, config)
    model.train(x_train)
else:
    weights_path = f"./model_weights/{GENERATIVE_MODEL}/{config.data.dataset}_{config.training.loss}.pth"
    model = utilities.get_model(GENERATIVE_MODEL, config, weights_path=weights_path)

# Test the model if required
model.test_generation(
    path=f"./results/{config.data.dataset}_{config.training.loss}",
    n_samples=16,
)
