import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def visualize_confusion_matrix(model, X, y):
    """ Display Confusion Matrix visually."""

    plot_confusion_matrix(model, X, y)
    plt.show()
    plt.close('all')

    return None
