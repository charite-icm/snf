import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt
import pytest
from unittest import mock
from src.snf_pipeline_revised import save_figure

@mock.patch("matplotlib.pyplot.tight_layout")
@mock.patch("matplotlib.pyplot.close")
@mock.patch("matplotlib.figure.Figure.savefig")
def test_save_figure_default(mock_savefig, mock_close, mock_tight_layout, capfd):
    """
    Test saving the figure with default parameters.
    """
    fig = plt.figure()
    fig_name = "test_figure"

    # Call the function
    save_figure(fig, fig_name)

    # Check that tight_layout is called
    mock_tight_layout.assert_called_once()

    # Check that fig.savefig is called with the correct file name and dpi
    mock_savefig.assert_called_once_with("test_figure.jpg", dpi=300)

    # Check the output for verbose=True
    captured = capfd.readouterr()
    assert "test_figure.jpg saved!" in captured.out

@mock.patch("matplotlib.pyplot.tight_layout")
@mock.patch("matplotlib.pyplot.close")
@mock.patch("matplotlib.figure.Figure.savefig")
def test_save_figure_multiple_formats(mock_savefig, mock_close, mock_tight_layout, capfd):
    """
    Test saving the figure with multiple formats.
    """
    fig = plt.figure()
    fig_name = "test_figure"
    img_formats = (".png", ".pdf")

    # Call the function
    save_figure(fig, fig_name, img_formats=img_formats)

    # Check that fig.savefig is called twice, once for each format
    assert mock_savefig.call_count == 2
    mock_savefig.assert_any_call("test_figure.png", dpi=300)
    mock_savefig.assert_any_call("test_figure.pdf", dpi=300)

    # Check the output for verbose=True
    captured = capfd.readouterr()
    assert "test_figure.png saved!" in captured.out
    assert "test_figure.pdf saved!" in captured.out

@mock.patch("matplotlib.pyplot.tight_layout")
@mock.patch("matplotlib.pyplot.close")
@mock.patch("matplotlib.figure.Figure.savefig")
def test_save_figure_no_verbose(mock_savefig, mock_close, mock_tight_layout, capfd):
    """
    Test saving the figure with verbose=False.
    """
    fig = plt.figure()
    fig_name = "test_figure"

    # Call the function with verbose=False
    save_figure(fig, fig_name, verbose=False)

    # Check that fig.savefig is called
    mock_savefig.assert_called_once_with("test_figure.jpg", dpi=300)

    # Check that no output is printed
    captured = capfd.readouterr()
    assert "saved!" not in captured.out

@mock.patch("matplotlib.pyplot.tight_layout")
@mock.patch("matplotlib.pyplot.close")
@mock.patch("matplotlib.figure.Figure.savefig")
def test_save_figure_plt_close(mock_savefig, mock_close, mock_tight_layout):
    """
    Test saving the figure with plt_close=True.
    """
    fig = plt.figure()
    fig_name = "test_figure"

    # Call the function with plt_close=True
    save_figure(fig, fig_name, plt_close=True)

    # Check that fig.savefig is called
    mock_savefig.assert_called_once_with("test_figure.jpg", dpi=300)

    # Check that plt.close is called
    mock_close.assert_called_once()

@mock.patch("matplotlib.pyplot.tight_layout")
@mock.patch("matplotlib.pyplot.close")
@mock.patch("matplotlib.figure.Figure.savefig")
def test_save_figure_custom_dpi(mock_savefig, mock_close, mock_tight_layout):
    """
    Test saving the figure with a custom dpi value.
    """
    fig = plt.figure()
    fig_name = "test_figure"
    custom_dpi = 600

    # Call the function with custom dpi
    save_figure(fig, fig_name, dpi=custom_dpi)

    # Check that fig.savefig is called with the custom dpi value
    mock_savefig.assert_called_once_with("test_figure.jpg", dpi=600)
