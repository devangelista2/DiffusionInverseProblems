model_params = {'min_signal_rate': 0.02,
                'max_signal_rate': 0.95,
                'input_shape': (64, 64, 1),
                'embedding_dims': 64,
                'embedding_max_frequency': 1000.0,
                'depths': [48, 96, 192, 384],
                'block_depth': 2}

training_params = {'batch_size': 100, 
                   'learning_rate': 0.0001,
                   'weight_decay': 1e-4,
                   'num_epochs': 50}

problem_params = {'data_path': './data/celeba_grayscale/',
                  'kernel_type': 'gaussian',
                  'k': 11,
                  'sigma': 1.3,
                  'noise_std': 0.02}

variational_params = {}