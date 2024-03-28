import matplotlib.pyplot as plt
import seaborn as sns
# Assuming scienceplots is correctly installed
# If not, you might need to install it using pip
import scienceplots

def configure_matplotlib(style='ieee', font_lang='en', seaborn_theme=False, font_scale=1.4):
    """
    Configure matplotlib and seaborn (optional) settings for plots.

    Parameters:
    - style: str, matplotlib style to use. Default is 'ieee'.
    - font_lang: str, language for font settings. 'en' for English (Times New Roman), 'cn' for Chinese (SimHei).
    - seaborn_theme: bool, whether to apply seaborn theme settings. Default is False.
    - font_scale: float, scaling factor for fonts if seaborn theme is applied.
    """
    # Define font settings
    fonts = {
        'en': {'family': 'Times New Roman', 'weight': 'normal', 'size': 12},
        'cn': {'family': 'simhei', 'weight': 'normal', 'size': 12},
    }

    # Apply matplotlib style
    plt.style.use(['science', style])

    # Configure fonts based on language
    if font_lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # To display Chinese characters correctly
    plt.rcParams['font.family'] = fonts[font_lang]['family']
    plt.rcParams['font.size'] = fonts[font_lang]['size']
    plt.rcParams['font.weight'] = fonts[font_lang]['weight']

    # Optionally apply seaborn theme
    if seaborn_theme:
        sns.set_theme(style="white", font='sans-serif', font_scale=font_scale)

if __name__ == '__main__':
    # Example usage
    configure_matplotlib(style='ieee', font_lang='en', seaborn_theme=False, font_scale=1.4)

