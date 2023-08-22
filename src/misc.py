# file to visualize images and json load function
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px



def data_clean_hist(img_dir):
    """
    Plot the histogram of an image file.

    Args:
        img_dir (str): Directory path of the image file.

    Returns:
        None
    """
    img = np.load(img_dir)
    plt.hist(img)
    plt.save()

def data_difficulty_hist(data, color='#EA5455', bgcolor='#F9F5EB', title='', xlabel='Score', ylabel='Count'):
    """
    Plots a histogram from the given data.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot 
        color (str): Color code for the histogram bars
        bgcolor (str): Background color for the plot
        title (str): Plot title 
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis

    Returns:
        fig (plotly Figure): The resulting histogram figure

    Raises:
        ImportError: If plotly is not installed

    """
    if px is None:
        raise ImportError("Plotly is not installed")

    fig = px.histogram(data, nbins=20, color_discrete_sequence=[color], opacity=0.85)
    
    fig.update_layout(
        plot_bgcolor=bgcolor, 
        xaxis_title=xlabel, 
        yaxis_title=ylabel,
        title=title,
        font=dict(
            family="Courier New, monospace",
            size=24
        ), 
        showlegend=False
    )
    
    return fig
# usage:
# import pandas as pd

# df = pd.DataFrame(...) # data

# fig = plot_histogram(df, title='My Histogram', ylabel='Number') 
# fig.show()