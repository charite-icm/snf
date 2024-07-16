import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-white")


# Save all images in these formats
# IMAGE_FORMATS = ["jpg", "png", "eps"]

IMAGE_FORMATS = [".jpg"]
DPI = 300

def save_figure(fig, fig_name: str, plt_close: bool = False) -> None:
    for format in IMAGE_FORMATS:
        plt.tight_layout()
        fig_full_name = fig_name + format
        fig.savefig(fig_full_name, dpi=300)
        print(f"{fig_full_name} saved!")
        if plt_close:
            plt.close()
