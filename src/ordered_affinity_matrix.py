import numpy as np
import matplotlib.pyplot as plt

from src.image_format import save_figure


def plot_ordered_affinity_matrix(network: np.ndarray,
                                 labels: list[int],
                                 figure_path: str,
                                 title: str = None,
                                 dynamic_range_th: tuple[float, float] = (0.1, 0.1),
                                 figsize: tuple[float, float] = (8.0, 8.0),
                                 show_colorbar: bool = False,
                                 plt_close: bool = True,
                                 dynamic_range: tuple[float, float] = None,
                                 return_dynamic_range: bool = False,
                                 show_axis: bool = False,
                                 high_quality: bool = False) -> None | tuple[float, float]:
    indexing_array = np.argsort(labels)
    fig, ax = plt.subplots(figsize=figsize)

    visualize_network = np.copy(network)
    np.fill_diagonal(visualize_network, 0)
    visualize_network /= np.nansum(visualize_network, axis=1, keepdims=True)

    max_sim = visualize_network.max()
    mean_sim = visualize_network.mean()

    np.fill_diagonal(visualize_network, 1)
    visualize_network_ordered = visualize_network[indexing_array][:, indexing_array]

    vmin, vmax = mean_sim - dynamic_range_th[0] * max_sim, mean_sim + dynamic_range_th[1] * max_sim
    if dynamic_range is not None:
        vmin, vmax = dynamic_range

    ax.imshow(visualize_network_ordered, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title, fontweight="bold")
    if show_colorbar:
        ax.colorbar()
    if not show_axis:
        ax.axis("off")

    if high_quality:
        save_plot_as_vector(fig, format="pdf", dpi=600, output_path=f"{figure_path}.pdf")
        save_plot_as_tiff(fig, column_type="double", dpi=600, output_path=f"{figure_path}.tiff")
    save_figure(fig, fig_name=f"{figure_path}.png", plt_close=plt_close)


    if return_dynamic_range:
        return vmin, vmax
