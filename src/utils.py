import matplotlib.pyplot as plt
import pickle


def plot_weights_distribution(model, layers_indices):
    """
    Plot distribusi bobot untuk layer-layer tertentu pada model.

      model: Instance dari FFNN.
      layers_indices: List indeks layer yang akan di-plot.
    """
    for idx in layers_indices:
        if idx < len(model.layers):
            plt.figure()
            plt.hist(model.layers[idx].W.flatten(), bins=30)
            plt.title(f"Distribusi Bobot - Layer {idx}")
            plt.xlabel("Nilai Bobot")
            plt.ylabel("Frekuensi")
            plt.show()


def plot_gradients_distribution(model, layers_indices):
    for idx in layers_indices:
        if idx < len(model.layers):
            plt.figure()
            plt.hist(model.layers[idx].dW.flatten(), bins=30)
            plt.title(f"Distribusi Gradien - Layer {idx}")
            plt.xlabel("Nilai Gradien")
            plt.ylabel("Frekuensi")
            plt.show()


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model
