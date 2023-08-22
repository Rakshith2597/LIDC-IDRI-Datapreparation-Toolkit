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

import numpy as np
import matplotlib.pyplot as plt

def data_diff_plot_uncertain_ratio(ratio, img_path, union_path, intersection_path, consensus_path):
    """
    Plot images based on uncertain ratio.
    
    Args:
        ratio (float): The uncertain ratio value
        img_path (str): Path to image folders
        union_path (str): Path to union image folders 
        intersection_path (str): Path to intersection image folders
        consensus_path (str): Path to consensus image folders
        
    Returns:
        None
        
    Raises:
        ImportError: If matplotlib is not installed
    """

    if plt is None:
        raise ImportError("Matplotlib is not installed")

    print(f'Uncertain Ratio: {ratio}')
    
    plt.subplot(141)
    plt.title('Union')
    im1 = np.load(img_path + '/' + ratio_dict[ratio][0]).astype(np.float)
    plt.imshow(im1, cmap='gray')
    
    plt.subplot(142) 
    plt.title('Union')
    im1 = np.load(union_path + '/' + ratio_dict[ratio][0]).astype(np.float)
    plt.imshow(im1, cmap='gray')

    plt.subplot(143)
    plt.title('Intersection')
    im2 = np.load(intersection_path + '/' + ratio_dict[ratio][0]).astype(np.float)
    plt.imshow(im2, cmap='gray')

    plt.subplot(144)
    plt.title('Consensus')
    im2 = np.load(consensus_path + '/' + ratio_dict[ratio][0]).astype(np.float)
    plt.imshow(im2, cmap='gray')

# example usage: 
# ratio = 0.9
# plot_images(ratio, img_path, union_path, intersection_path, consensus_path)