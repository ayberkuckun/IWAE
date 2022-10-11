import numpy as np


def find_active_units(encoder, data, stochastic_layer_count):
    if stochastic_layer_count == 1:
        z_mean_1, _, _ = encoder(data)

        variances_1 = np.var(z_mean_1, axis=0)
        active_neurons_1 = np.where(variances_1 > 1e-2, 1, 0)
        active_neuron_count_1 = np.count_nonzero(active_neurons_1)

        print(f"Active units: {active_neuron_count_1}")

    elif stochastic_layer_count == 2:
        z_mean_1, z_mean_2, _, _ = encoder(data)

        variances_1 = np.var(z_mean_1, axis=0)
        active_neurons_1 = np.where(variances_1 > 1e-2, 1, 0)
        active_neuron_count_1 = np.count_nonzero(active_neurons_1)

        variances_2 = np.var(z_mean_2, axis=0)
        active_neurons_2 = np.where(variances_2 > 1e-2, 1, 0)
        active_neuron_count_2 = np.count_nonzero(active_neurons_2)

        print(f"Active units: {active_neuron_count_1} + {active_neuron_count_2}")
