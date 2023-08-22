# file to visualize images and json load function
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import pylidc as pl
from pylidc.utils import consensus



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


def data_diff_visualize_nodule(series_uid, clevel=0.5, pad=[(20,20),(20,20),(0,0)]):
  """Visualize lung nodule annotations and consensus contour for a given CT scan.

  Args:
    series_uid (str): Series instance UID of the CT scan
    clevel (float): Consensus level threshold  
    pad (list): Padding to add context

  Returns:
    None
  
  Raises:
    ImportError: If matplotlib/pylidc is not installed
  """

  if plt is None or pl is None:
    raise ImportError("Matplotlib and pylidc must be installed")

  scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
  vol = scan.to_volume()
  nods = scan.cluster_annotations()

  print(f"Number of nodule clusters: {len(nods)}")

  for i in range(len(nods)):
    anns = nods[i]  
    cmask, cbbox, masks = consensus(anns, clevel=clevel, pad=pad)
    k = int(0.5*(cbbox[2].stop - cbbox[2].start))

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)

    colors = ['r', 'g', 'b', 'y']
    for j in range(len(masks)):
      for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
         label = f"Annotation {j+1}"
         plt.plot(c[:,1], c[:,0], colors[j], label=label)

    for c in find_contours(cmask[:,:,k].astype(float), 0.5):   
      plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')

    ax.axis('off')
    ax.legend()
    plt.tight_layout() 
    plt.show()

# example usage: 
# series_uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178' 
# visualize_nodule(series_uid)


def data_case_wise_list_plot_variability_count(case_data, colors=None):
  """
  Plots a bar chart showing count of different variability cases.

  Args:
    case_data (dict): Dictionary with case names as keys and counts as values
    colors (list): List of colors for each bar

  Returns: 
    fig: matplotlib figure object

  Raises:
    ImportError: If matplotlib is not installed

  """

  if plt is None:
    raise ImportError("Matplotlib required for plotting")

  cases = list(case_data.keys())
  values = list(case_data.values())

  if not colors:
    colors = ['red', 'yellow', 'blue']

  fig = plt.figure(figsize=(10, 5))
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.bar(cases, values, color=colors)
  plt.xlabel("Type of variability", fontsize=28)
  plt.ylabel("Count", fontsize=28)
  
  return fig

#example usage:
# data = {'Case 0': 10, 'Case 1': 15, 'Case 2': 20}
# fig = plot_variability_count_chart(data)
# plt.show()