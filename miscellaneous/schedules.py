import tensorflow as tf

import config

# Load configs
min_signal_rate = config.model_params['min_signal_rate']
max_signal_rate = config.model_params['max_signal_rate']

def improved_cosine(diffusion_times):
    # diffusion times -> angles
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    # angles -> signal and noise rates
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates