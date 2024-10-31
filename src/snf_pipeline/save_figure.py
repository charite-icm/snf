import matplotlib.pyplot as plt


def save_figure(fig, fig_name: str, plt_close: bool = False, img_formats: tuple[str] = (".jpg", ),
                dpi: int = 300, verbose: bool = True) -> None:
    """
    Save a matplotlib figure in one or more image formats and optionally close the plot.

    This function saves the given `fig` (a matplotlib figure) to the file system in the specified image format(s) with
    the given DPI (dots per inch) resolution. The function can also optionally close the figure after saving it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.

    fig_name : str
        The base file name (without extension) for the figure. The file extension(s) will be determined by `img_formats`.

    plt_close : bool, optional
        Whether to close the figure after saving. If `True`, `plt.close()` will be called. Default is `False`.

    img_formats : tuple of str, optional
        A tuple of strings specifying the file formats in which to save the figure (e.g., ".jpg", ".png"). 
        Default is `(".jpg",)`.

    dpi : int, optional
        The resolution in dots per inch for the saved figure. Default is 300.

    verbose : bool, optional
        Whether to print a message after each file is saved. If `True`, a message will be printed. Default is `True`.

    Raises
    ------
    None

    Returns
    -------
    None
        The function saves the figure to the specified file format(s) and optionally closes the plot.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> save_figure(fig, "plot", img_formats=(".png", ".pdf"), dpi=200, verbose=True, plt_close=True)

    This will save the figure as "plot.png" and "plot.pdf" with 200 DPI, print a message for each saved file, and 
    close the figure after saving.
    """
    for img_format in img_formats:
        plt.tight_layout()
        fig_full_name = fig_name + img_format
        fig.savefig(fig_full_name, dpi=dpi)
        if verbose:
            print(f"{fig_full_name} saved!")
        if plt_close:
            plt.close()