#!/usr/bin/env python
# coding: utf-8

# # Assignment 3. YouTube Dashboard

# # Overview
# For this assignment, I wanted to dig into the world of YouTube and downloaded the Kaggle dataset called <b>Trending YouTube Video Statistics</b>, available at [Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new). This is a great public dataset for the media world, which is the industry I work in.
# 
# Here are some interesting highlights about this dataset:
# * <b>Daily trending data</b>: I wanted to ensure that the dataset was rich enough to do interesting stuff with. The data captures the top ~200 trending YouTube videos each day across multiple countries, including titles, channel names, publication timestamps, category IDs, tags, view counts, likes/dislikes, comment counts, and more
# * <b>Multi-country coverage</b>: I figured it would be interesting to take a look at behavior across countries. This dataset includes datasets for regions like the USA, Great Britain, Canada, India, Japan, etc., each maintained separately—facilitating both local and global trend analysis. Though I eventually realized that the most interesting cut would be to analyze trends across countries with the same primary language -- so I limited the data analysis to US, Canada and Great Britain (unfortunately, Australia wasn't available)
# * <b>Rich metadata for each video</b>: I wanted to be able to look at both aggregate trends, but drill into individual data points when needed, say, with a scatterplot. In addition to numerical metrics (views, likes, dislikes, comments), it provides descriptive data such as the video title, channel name, publish time, and tags—allowing for both quantitative and qualitative insights
# * <b>Trends over time</b>: I really wanted to do something longitudinal for the line chart. Because videos often trend for multiple days, the dataset enables tracking of prominence trajectories, e.g., how long a video remains trending or how engagement evolves day by day.
# 
# It's an intriguing dataset that lets us dig into the characteristics of videos that go viral on YouTube. Let's dive in!

# ---
# # Installations
# 
# There was a lot of experimentation and research I did before deciding on the library I wanted to use for this assignment. I wanted a set of libraries that have been proven to work within the Vocareum JupyterLab environment -- since I know the university has had its share of issues with it. 
# 
# Since we are not suppose to use Matplotlib or Altair for this, I primarily weighed between Plotly and Seaborn. I read that Seaborn was primarily static, and wasn't as easy to integrate with interactive widget controls from ipywidget or Panel, and I really wanted to try my hand at an altogether unfamiliar library, so I decided to go with Plotly for charting.
# 
# For dashboarding and interactive widgetry, I went with Panel. Panel allowed for finer-grained control and had a more modern approach compared to ipywidget, and also read that the Bokeh integration was seamless, since both Panel and Bokeh were produced by the same team. Bokeh is the Python interactive visualization library that powers tools like Panel, giving us rich browser-based visualization (something I unfortunatey had trouble getting working within the Vocareum environment) and dashboards. 
# 
# ## Library Summary
# So in summary, I went with
# * Plotly for Static Charts
# * Panel for Dashboarding
# * Panel for interactive widgetry
# 
# <b>Plotly 5.18.0</b>: This version of Plotly is considered very stable and widely used. To be honest, I started with Plotly 6.x, but found that it wasn't rendering anything in the notebook, even though it didn't throw any errors. Newer versions were also running into dependency issues with Kaleido, which is used by Plotly to export static images (.jpgs, .pngs, ...) from interactive figures, and I could simply not set it all up to work within Vocareum (something to raise with the Vocareum team!)
# * **Developed by**: [Plotly Inc](https://plotly.com/)
# * **Open source**? Yes, under the MIT license
# * **General Approach**: a Javascript plotting library with Python bindings
# * **Type**: Declarative
# * **Jupyter Integrations**: Yes, native integrations that work with classic Notebooks, JupyterLab and CoLab
# * **Advantages**: Charting felt a lot more feature-rich, with default filtering mechanisms and rich hoverstates (I'll demo that later)
# * **Limitations**: Image exports require kaleido, and more modern varsions had compatibility issues within Vocareum
# 
# 
# <b>Panel</b>: Panel was covered in some of the more advanced videos, and I wanted to play around with it. As stated, we'll need Panel for the widgetry and dashboarding. The default styling also doesn't seem as slick as the others
# * **Developed by**: [Holoviz.org](https://holoviz.org/)
# * **Open source**: Yes, under the BSD 3-Clause license
# * **General approach**: a dashboarding framework for Python, designed to work with a wide range of plotting libraries
# * **Type**: Declarative when using the @pn.depends decorator or pn.bind, procedural for callbacks such as .on_click()
# * **Jupyter Integrations**: Yes, integrates well with Classic Notebooks and JupyterLab (requires pn.extension())
# * **Advantages**: Can view dashboards inline or as standalone web apps. Has a rich layout and composition system and works well for both reactive (auto-updating) and event-driven(submit buttons). Thankfully! After building my dashboard, I found the reactive version too slow, and switched to the event-driven model
# * **Limitations**: Harder to learn versus ipywidgets. The decorator and .bind() operation wasn't the most intuitive.
# 
# 
# <b>ipywidgets</b>: Even if I am using Panel, I still needed to import the ipywidget library because Jupyter Notebook's front end requires it to render interactive widgets reliably, even if we're not explicitly using it in the code.
# * **Developed by**: Jupyter Project, maintained by core contributors of Project Jupyter
# * **Open source**: Yes, under the BSD license
# * **General approach**: provides interactive widgets for controlling Python code in Jupyter notebooks
# * **Type**: Procedural -- callback functions are attached manually (.observe, .on_click, ...)
# * **Jupyter Integrations**: Yes, it is THE native widgeting system in Jupyter
# * **Advantages**: Built by the Jupyter guys, so it's native, lightweight and simple.
# * **Limitations**: Weak layout and composition system. It doesn't scale well for more complex dashboards or cross-library integrations
# 
# 

# In[1]:


#!pip install plotly ipywidgets
#!pip install plotly==5.18.0
#!pip install jupyterlab_widgets ipywidgets ipykernel --upgrade


# # Import and Setup
# 
# Here are all the libraries I used (including some that I brought in for debugging, such as pprint)

# In[2]:


import pandas as pd
import numpy as np
import os
import json
import pprint

import plotly
import plotly.io as pio   # We're going to use Plotly for the charts
import plotly.graph_objs as go

import panel as pn        # We're going to use Panel for dashboarding

# Set default renderer for Plotly figures in notebooks.
# "iframe" ensures isolated, static rendering (more stable across environments),
# while "notebook_connected" enables dynamic interaction but can fail in restricted environments like Vocareum.
pio.renderers.default = "iframe"  

#print(pn.__version__)
pd.set_option('display.float_format', '{:.5f}'.format)   # I don't like Panda's default usage of scientific notation


# ⚠️ Ignore the warning above.
# I didn’t — and upgrading Plotly to 6.1.1 (as the warning suggested) introduced new issues that broke Panel compatibility and caused silent failures when rendering Plotly figures inside my dashboard. So despite the Kaleido warning, I stayed with Plotly 5.18.0, which works reliably with kaleido==0.2.1 in environments like Vocareum. Static image export isn’t a priority for this notebook — interactive exploration is.
# 
# I then downloaded the .zip file from [Kaggle's Trending YouTube Video Statistics page](https://www.kaggle.com/datasets/datasnaek/youtube-new) into the folder tree. The function below unzips it into a directory called youtube_data/. To speed up execution, I use the os library to detect the presence of the folder and skip the unzipping. 

# In[3]:


import os

def unzip_files(file_path = "youtube_archive.zip", force=False):
    """
    Unzips a YouTube dataset archive if it hasn't already been extracted.

    Parameters:
    -----------
    file_path : str, optional (default="youtube_archive.zip")
        Path to the ZIP file containing the dataset.

    force : bool, optional (default=False)
        If True, will re-extract the archive even if the target directory exists.
        (Note: force behavior is not yet implemented in this version.)

    Behavior:
    ---------
    - If the "youtube_data/" directory already exists, the function does nothing.
    - If the ZIP file exists, it prints basic info about it.
    - If neither the directory nor the ZIP file is found, it prints an error message.

    Returns:
    --------
    None
    """
    if os.path.exists("youtube_data/"):
        print("Directory already exists")
        return    # No need to do anything since it's already been unzipped
    elif os.path.exists(file_path):
        print(f"File found: {file_path}")
        print(f"Size: {os.path.getsize(file_path) / 1024:.2f} KB")
    else:
        print("File not found.")

    return

unzip_files()


# # Country Inspections
# I limited analysis to the US, Canda and Great Britain because they all share a common language, and I felt that behavioral comparisons of those videos would make the most sense. I also considered grouping them by the actual video and looking at only English-native countries felt like they would have the greatest overlap. I eventually considered not doing that because setting up the basic dashboard was already enough work! Nevertheless, it's an opportunity for future analysis.
# 
# Let's inspect the US dataset to see if there's any funny business.

# In[4]:

 
# # Loading the Data into our master_df
# 
# Now it's time to load the data from the three .csv files and corresponding .json files. Let's create the two extraction functions below. I put a whole bunch of debugging statements in there, which can be accessed by setting **verbose** to True.

# In[8]:


# Now that we're done with the visual inspection, let's genericize the loading call
def load_yt_country_data(country:str, verbose=False):
    """
    Loads the trending YouTube video data for a specific country.

    Parameters:
    -----------
    country : str
        Two-letter country code (e.g., "US", "GB", "IN").
        The function will look for a file named '{country}videos.csv' under the 'youtube_data/' directory.

    verbose : bool, optional (default=False)
        If True, prints data type information, summary statistics, and sample rows.

    Behavior:
    ---------
    - Reads the CSV file using Latin-1 encoding (used in YouTube trending datasets).
    - Parses 'publish_time' and converts 'trending_date' from its custom format (%y.%d.%m) to a proper datetime.
    - Optionally displays schema and samples if verbose=True.

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame for the specified country's trending YouTube videos.
    """

    df = pd.read_csv(f"youtube_data/{country}videos.csv", encoding="latin1", parse_dates=["publish_time"],nrows=20000)

    # Drop the tags, thumbnail_link and description to save on memory
    df.drop("tags", inplace=True, axis=1)
    df.drop("thumbnail_link", inplace=True, axis=1)
    df.drop("description", inplace=True, axis=1)

    # Trending_date is in a weird format. Will need to parse that after TODO
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')

    if verbose:
        print(df.dtypes)
        print(df.info())
        print(df.describe())
        print(us_df.sample(5))

    return df

def load_category_json(country="US", verbose=False):
    """
    Loads and processes the category ID JSON file for a given country.

    Parameters:
    -----------
    country : str, optional (default="US")
        Two-letter country code used to locate the '{country}_category_id.json' file.

    verbose : bool, optional (default=False)
        If True, displays the raw and parsed category structure for debugging.

    Behavior:
    ---------
    - Parses the JSON file and extracts a dictionary mapping category IDs to human-readable titles.
    - Applies a manual fix to distinguish between two ambiguous "Comedy" categories:
        * ID 23 → 'Comedy (UGC)'
        * ID 34 → 'Comedy (Movies)'

    Returns:
    --------
    dict
        A dictionary mapping integer category IDs to descriptive titles.
    """
    with open(f"youtube_data/{country}_category_id.json") as fh:
        raw_categories = json.load(fh)

    if verbose:
        print(type(us_raw_categories), us_raw_categories)

    # Create a simplified dictionary of categories
    categories = {int(item['id']): item['snippet']['title']  
        for item in raw_categories['items']}

    # Ugh, YouTube category structures contains two 'Comedy' categories. The earlier (23) refers to UGC
    # while the latter refers to Movies/Professional Content. We should fix that manually.

    categories[23] = 'Comedy (UGC)'
    categories[34] = 'Comedy (Movies)'

    if verbose:
        print(f"{country} Categories", categories)

    return categories

#load_category_json(verbose=True)


# Let's pull it all together with one master function to load all the .csv data, load all the .json data, merge them, then perform all the necessary wrangling to make the visualizations more useful.
# 
# Once I started visualizing, I found that a large number of metrics were right-skewed, so we had to augment the dataframe with a couple of np.log10 (and np.log1p) transformation functions.

# In[9]:


def load_and_augment_yt_data(verbose=False):
    """
    Loads, merges, and augments trending YouTube data for multiple countries.

    Parameters:
    -----------
    verbose : bool, optional (default=False)
        If True, displays the shape, schema, and a sample of the resulting DataFrame.

    Behavior:
    ---------
    - Loads trending video data for the US, Canada, and Great Britain.
    - Adds a 'country' column to distinguish rows from each source.
    - Maps `category_id` to descriptive category names using the US category JSON file
      (since CA and GB use a subset of US categories).
    - Converts category names to pandas Categorical type for memory efficiency.
    - Adds ratio-based engagement metrics:
        * likes/views
        * dislikes/views
        * comments/views
    - Computes `days_to_trend` as the difference between `trending_date` and `publish_time`
      (with -1 values corrected to 0).
    - Applies log transformations to skewed metrics:
        * log_views, log_likes, log_dislikes, log_comment_count
        * log_days_to_trend (using log1p for numeric stability)

    Returns:
    --------
    pd.DataFrame
        A fully processed and augmented DataFrame combining US, CA, and GB trending video data.
    """
    countries = ['US', 'CA', 'GB']
    master_df = pd.DataFrame()
    for country in countries:
        tempdf = load_yt_country_data(country)
        tempdf['country'] = country

        master_df = pd.concat([master_df, tempdf])

    # For categories, we can just load the US one, since we've confirmed that CA and GB are a subset
    categories = load_category_json()

    # Let's map the categories
    master_df['category'] = master_df['category_id'].map(categories)
    # Let's convert it into pd.Categorical for memory efficiency
    categoryList = pd.CategoricalDtype(categories=categories.values())
    master_df['category'] = master_df['category'].astype(categoryList)

    # Now, let's add columns that compute the ratios
    master_df['likes/views'] = (master_df['likes']/master_df['views'])
    master_df['dislikes/views'] = (master_df['dislikes']/master_df['views'])
    master_df['comments/views'] = (master_df['comment_count']/master_df['views'])

    # Let's also compute the days to trend
    master_df['days_to_trend'] = (master_df['trending_date'] - 
                                  master_df['publish_time'].dt.tz_localize(None)).dt.days

    # A few of these show up as -1 because of idiosyncracies in the timezone. Replace with 0
    master_df['days_to_trend'] = master_df['days_to_trend'].replace(-1, 0)
    master_df['log_days_to_trend'] = np.log1p(master_df['days_to_trend'] + 1)      # Seems that there are a few outliers in this as well

    # Transformations: All the standard YouTube metrics are VERY right-skewed, with 
    # certain videos receiving MASSIVE viewership. I believe all the standard metrics
    # need to be transformed, since they look awful in the boxplots
    for col in ['views', 'likes', 'dislikes', 'comment_count']:
        master_df['log_' + col] = np.log10(master_df[col] + 1)

    if verbose:
        # For US + CA + GB, we expect 120,746
        print(f"Shape: {master_df.shape}")
        print(master_df.dtypes)
        print(master_df.sample(5))

    return master_df



# Finally, let's define master_df at the top-level since it will be the primary dataframe we use for all our visualization functions!

# In[10]:


# Set master_df:
master_df = load_and_augment_yt_data(verbose=True)


# ## Category Cleanup
# 
# Now that we joined the .csv data with the .json category file, I was curious to see what the distribution of categories were. We can easily do that with an untruncated pivot table.

# In[11]:

# There is a surprising number of 0s in the pivot table. For the purposes of keeping our visualization controls simple, we will limit the valid category list to categories that have non-zero values across all three countries. I guess certain categories don't ever go viral...

# In[12]:


valid_categories = ['Film & Animation', 
                    'Autos & Vehicles',
                    'Music',
                    'Pets & Animals',
                    'Sports',
                    'Travel & Events',
                    'Gaming',
                    'People & Blogs', 
                    'Comedy (UGC)',
                    'Entertainment',
                    'News & Politics',
                    'Howto & Style',
                    'Education',
                    'Science & Technology',
                    'Nonprofits & Activism',
                    'Shows']

# Sanity check
missing = [cat for cat in valid_categories if cat not in master_df['category'].unique()]
print("Missing:", missing)



# # Planning our Charts
# 
# Before embarking on the visualizations (and even some of the data augmentation), I spent some time "Planning" what would be contained in the dashboard. I wrote the following to capture what I was hoping to achieve with the exploratory visual work. "Planning" like this was certainly an extremely helpful step before jumping into coding:
# 
# >I'm trying to think about all the interesting questions I can ask about this dataset for a dashboard. Here's what I originally sketched out:
# >
# >**Histogram**
# >1) (Without transformation) # of views, # of likes, # of dislikes, # of comments
# >2) (Ratios scaled to % of views): % of likes, % of dislikes, % of comments
# >3) (Aggregations) # of trending videos published,  filtered by all or by category
# >
# >**Boxplot**
# >1) (Without transformation) # of views, # of likes, # of dislikes, # of comments with country on the X-axis
# >2) (Ratios scaled to % of views): % of likes, % of dislikes, % of comments with country on the X-axis
# >3) (Aggregations) # of trending videos published,  filtered by all or by category with country on the X-axis
# >
# >**Scatterplot**
# >1) (Without transformation) Various combos of # of views, # of likes, # of dislikes, # of comments 
# >2) (Ratios scaled to % of views): % of likes, % of dislikes, % of comments
# >3) (Aggregations) Various combos of # of views, # of likes, # of dislikes, # of comments but aggregated by category
# >
# >**Line Graph**
# >1) (Without transformation)  # of views, # of likes, # of dislikes, # of >comments  over time
# >2) (Ratios scaled to % of views): % of likes, % of dislikes, % of comments over time
# >3) (Aggregations) Trend of different categories' views/likes/dislikes/comments over time
# >
# >I suspect there won't be as much overlap across different language countries, so maybe I should limit the analysis to US, Canada and GB. I can keep those three countries as colored hues (or in the case of boxplots, as separate categories on the x-axis). I'm also thinking that these are two dashboards -- one working off of the base data (without transformation or the extra columns I add to scale it to the country's population), and a completely separate dashboard for the aggregations.
# >
# >Other metrics:
# >* Ratios: likes/views, dislikes/views, comments/views
# >* Days to trend
# 
# Looking back at this plan, I'm pleased to say that the dashboard could successfully address all the questions I had originally scoped! There were a few changes I had to make -- such as switching to log transformations for most of the metrics when I was displeased with the visuals for the pure metrics, or switching out the Boxplot for a more expressive ViolinPlot.
# 
# I had originally thought that I would have had to create another dashboard for the 3) (Aggregation) questions -- but was delighted to find that it wasn't necessary!

# # Plotting each dashboard without Interactivity
# 
# So, as I mentioned above, since I've been playing around with matplotlib for the previous two weeks, I wanted to try out another plotting library (plus, the Assignment expressly forbids using Matplotlib or Altaire!). I was going to build these widgets using ipywidgets and Seaborn, but I read somewhere that they don't interact very well. HVPlot seems too much of a jump, and doesn't work well on this Vocareum Jupyter environment (we gotta download Jupyter Notebooks into our local machine). We played around with Altair all of last month in 522, so I settled to try out Plotly.
# 
# First, each of our charting options does some validation to make sure we have valid entries. So let's set some variables to list out the acceptable values. We'll use these to also control the interactive widgets later on.
# 

# In[13]:


# We need to list the valid countries that can be passed in as an argument 
# (specifically for histograms and line plots. Violin and scatter can render all three of them easily)
# In addition, the scatterplot needs an associated shape with them
country_symbol_map = {'US': 'circle', 'CA': 'square', 'GB': 'diamond'}

# All charts require us to pass a metric to pivot the Y-axis off of. In addition, the scatterplot
# requires one of these metrics for its x-axis as well
valid_metrics = ['log_views','log_likes','log_dislikes','log_comment_count',
                 'likes/views','dislikes/views','comments/views',
                 'days_to_trend', 'log_days_to_trend']

# The line chart can pivot its x-axis on either the YouTube publish date, or the date it started trending
dateOptions = ['trending_date', 'publish_time']

# Time Rollups are used to control the line charts
valid_time_rollups = {'Daily':'D', 
                      'Weekly':'W', 
                      'Monthly':'M'}

# Line charts CAN in theory, display the non-log10 versions of views, likes, dislikes, comment_count. 
# We also need to map out the corresponding aggregation function of each for the lineplot
expanded_metrics_aggs = {'views': 'sum',
                         'likes': 'sum',
                         'dislikes': 'sum',
                         'comment_count': 'sum',
                         'log_views': 'mean',
                         'log_likes': 'mean',
                         'log_dislikes': 'mean',
                         'log_comment_count': 'mean',
                         'likes/views': 'mean',
                         'dislikes/views': 'mean',
                         'comments/views': 'mean',
                         'days_to_trend': 'mean',
                         'log_days_to_trend': 'mean'}


# ## Boxplot
# Let's first start out with boxplots, the most basic of the charts! 

# In[14]:


# Let's start playing around with Plotly. Let's first try boxplots

def plot_boxplot(df, y_axis, catFilters=None, show=True):
    """
    Creates a Plotly boxplot showing the distribution of a selected metric across countries.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing YouTube video data, including 'category', 'country',
        and the metric specified in `y_axis`.

    y_axis : str
        The name of the numeric column to use for the Y-axis (e.g., 'log_views', 'log_likes').
        Must be one of the predefined `valid_metrics`.

    catFilters : list of str, optional (default=None)
        A list of category names to filter the data before plotting.
        If None, the full dataset is used.

    show : bool, optional (default=True)
        If True, displays the figure immediately using `fig.show()`.
        If False, returns the Plotly figure object (useful for embedding in dashboards).

    Behavior:
    ---------
    - Filters the DataFrame by category if `catFilters` is provided.
    - Groups the data by country and plots one box trace per country.
    - Displays individual outlier points on each boxplot.
    - Sets a consistent layout and figure height.

    Returns:
    --------
    None or plotly.graph_objects.Figure
        Returns the Plotly Figure object if `show=False`, otherwise displays the chart inline.
    """

    # Error-check values
    if y_axis not in valid_metrics:
        raise ValueError(f"Invalid value {y_axis}. Y-axis needs to be one of {valid_metrics}")

    fig = go.Figure()
    fig.update_layout(height=400)  # set height to be consistent

    if catFilters is not None:
        print("catFilters", catFilters)
        cat_subset_df = df[df['category'].isin(catFilters)]
    else: 
        print("No catFilters")
        cat_subset_df = df.copy()

    for country in cat_subset_df['country'].unique():
        subset_df = cat_subset_df[cat_subset_df['country'] == country]

        fig.add_trace(go.Box(
            y=subset_df[y_axis],
            name=country,
            boxpoints='outliers',  # Show individual outlier points
            marker=dict(opacity=0.6)
        ))

    fig.update_layout(
        title=f'Boxplot: {y_axis} distribution by country',
        yaxis_title = y_axis,
        xaxis_title = "Country",
        showlegend = False,
        height = 600
    )
    if show:
        fig.show()
    else:
        return fig





# In[15]:


# Let's take a look at the boxplot
# 👇 Feel free to play with the variables below to see how the chart changes!

# Y-axis metric (valid options):
# ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
#  'likes/views', 'dislikes/views', 'comments/views',
#  'days_to_trend', 'log_days_to_trend']
metric = 'likes/views'

# Category filter (optional); set to None for all, or use a subset of:
# ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals',
#  'Sports', 'Travel & Events', 'Gaming', 'People & Blogs', 'Comedy (UGC)',
#  'Entertainment', 'News & Politics', 'Howto & Style', 'Education',
#  'Science & Technology', 'Nonprofits & Activism', 'Shows']
catFilters = None
#catFilters = ['Music', 'Sports', 'Gaming']
plot_boxplot(master_df, metric, catFilters)


# The logs made matrics like views, likes, dislikes, and comment_count more balanced. But I didn't expect the ratios dislikes/views and comments/views to also be right-skewed. However, I'm not a big fan of taking the log of derived ratios, so we'll keep those as-is!
# 
# ## Violinplot
# I'm also a bigger fan of Violinplots ever since I got introduced to them. They are so much more expressive than Boxplots and I want to use a ViolinPlot for the Dashboard instead of a Boxplot.
# 
# The violinplot takes a metric, a category list of filters (None, if you want to show all categories). For fun, we also enabled the hoverstate for the outliers so you can see the video and channel.

# In[16]:


def plot_violinplot(df, y_axis, catFilters=None, show=True):
    """
    Generates a violin plot to visualize the distribution of a selected metric
    across countries, optionally filtered by content category.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing YouTube video data, including columns
        like 'category', 'country', the target metric, 'title', and 'channel_title'.

    y_axis : str
        The name of the numeric column to use for the Y-axis.
        Must be one of the values in `valid_metrics`:
        ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
         'likes/views', 'dislikes/views', 'comments/views',
         'days_to_trend', 'log_days_to_trend']

    catFilters : list of str, optional (default=None)
        A list of category names to filter the dataset before plotting.
        If None, includes all categories.

    show : bool, optional (default=True)
        If True, displays the plot immediately.
        If False, returns the Plotly figure object.

    Behavior:
    ---------
    - Filters the data by category if `catFilters` is provided.
    - Groups the filtered data by country and adds a separate violin trace per country.
    - Each violin includes an embedded boxplot and mean line.
    - Hover tooltips show the category, value, video title, and channel name.

    Returns:
    --------
    None or plotly.graph_objects.Figure
        Displays the figure if `show=True`, otherwise returns it.
    """
    # Error-check values
    if y_axis not in valid_metrics:
        raise ValueError(f"Invalid value {y_axis}. Y-axis needs to be one of {valid_metrics}")

    fig = go.Figure()
    fig.update_layout(height=400)  # set height to be consistent

    title = f'Violinplot: {y_axis} distribution by country'
    if catFilters is not None:
        cat_subset_df = df[df['category'].isin(catFilters)]
        cat_list = ', '.join(catFilters)
        if len(catFilters) > 4:
            cat_list = ', '.join(catFilters[:4]) + f', +{len(catFilters) - 4} more'
        title += f'<br><sup>Categories: {cat_list}</sup>'
    else: 
        cat_subset_df = df.copy()
        title += "<br><sup>No category filters</sup>"

    for country in cat_subset_df['country'].unique():
        subset_df = cat_subset_df[cat_subset_df['country'] == country]

        customdata = subset_df[['title', 'channel_title']]
        fig.add_trace(go.Violin(
            y=subset_df[y_axis],
            name=country,
            box_visible=True,    # Adds an embedded boxplot
            meanline_visible = True,   # Adds a mean line
            points='outliers',  # Show individual outlier points
            marker=dict(opacity=0.6),
            customdata=customdata,
            hovertemplate=(
                '<b>Category:</b> %{x}<br>' +
                '<b>Value:</b> %{y}<br>' +
                '<b>Title:</b> %{customdata[0]}<br>' +
                '<b>Channel:</b> %{customdata[1]}' +
                '<extra></extra>'
                ),
        ))

    fig.update_layout(       
        title=f'Violinplot: {y_axis} distribution by country',
        yaxis_title = y_axis,
        xaxis_title = "Country",
        showlegend = False,
        height = 600
    )
    if show:
        fig.show()
    else:
        return fig



# In[17]:


# Let's take a look at the violinplot
# 👇 Feel free to play with the variables below to see how the chart changes!

# Y-axis metric (valid options):
# ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
#  'likes/views', 'dislikes/views', 'comments/views',
#  'days_to_trend', 'log_days_to_trend']
metric = 'log_comment_count'

# Category filter (optional); set to None for all, or use a subset of:
# ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals',
#  'Sports', 'Travel & Events', 'Gaming', 'People & Blogs', 'Comedy (UGC)',
#  'Entertainment', 'News & Politics', 'Howto & Style', 'Education',
#  'Science & Technology', 'Nonprofits & Activism', 'Shows']
catFilters = None
#catFilters = ['Sports', 'Gaming']
plot_violinplot(master_df, metric, catFilters)


# # Histogram
# 
# Let's now try doing a histogram in plotly. The histogram takes two filters: country and category. If either or both are set to None, then no filter will be applied. You can also passed the desired number of bins (otherwise, it will default to 50)

# In[18]:


def plot_histogram(df, y_axis, bins=50, countryFilters = None, catFilters=None, show=True):
    """
    Plots a histogram of a specified metric, optionally filtered by country and category.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing YouTube video data, including columns like
        'category', 'country', and the specified `y_axis` metric.

    y_axis : str
        The name of the numeric column to plot on the X-axis (binned).
        Must be one of the `valid_metrics`:
        ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
         'likes/views', 'dislikes/views', 'comments/views',
         'days_to_trend', 'log_days_to_trend']

    bins : int, optional (default=50)
        The number of bins to use in the histogram.

    countryFilters : list of str, optional (default=None)
        A list of country codes (e.g., ['US', 'CA']) to filter the data.
        If None, includes all countries.

    catFilters : list of str, optional (default=None)
        A list of category names to filter the data.
        If None, includes all categories.

    show : bool, optional (default=True)
        If True, displays the figure immediately.
        If False, returns the Plotly figure object for embedding.

    Behavior:
    ---------
    - Applies optional filtering by country and category.
    - Plots a histogram using Plotly with the specified number of bins.
    - Adds informative title text showing the active filters.

    Returns:
    --------
    None or plotly.graph_objects.Figure
        Displays the figure if `show=True`; otherwise returns it for use in a dashboard.
    """
    # Error-check values
    if y_axis not in valid_metrics:
        raise ValueError(f"Invalid value {y_axis}. Y-axis needs to be one of {valid_metrics}")

    fig = go.Figure()
    fig.update_layout(height=400)  # set height to be consistent

    title = f'Histogram: {y_axis}'
    if countryFilters is not None:
        country_subset = df[df['country'].isin(countryFilters)]
        title = title + f'<br><sup>Country filters: {countryFilters}</sup>'
    else:
        country_subset = df.copy()    
        title = title + '<br><sup>No country filters</sup>'

    if catFilters is not None:
        subset_df = country_subset[country_subset['category'].isin(catFilters)]
        cat_list = ', '.join(catFilters)
        if len(catFilters) > 4:
            cat_list = ', '.join(catFilters[:4]) + f', +{len(catFilters) - 4} more'
        title += f'<br><sup>Categories: {cat_list}</sup>'

    else:
        subset_df = country_subset.copy()
        title = title + '<br><sup>No category filters</sup>'

    fig.add_trace(go.Histogram(
        x = subset_df[y_axis],
        nbinsx = bins    
    ))


    fig.update_layout(       
        title = title,
        yaxis_title = y_axis,
        xaxis_title = "Bins",
        showlegend = False,
        height = 600
    )
    if show:
        fig.show()
    else:
        return fig


# In[19]:


# Let's take a look at the histogram
# 👇 Feel free to play with the variables below to see how the chart changes!

# Y-axis metric (valid options):
# ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
#  'likes/views', 'dislikes/views', 'comments/views',
#  'days_to_trend', 'log_days_to_trend']
metric = 'log_views'

# Country filter (subset of or None):
# ['US', 'CA', 'GB']
#countryFilters = None
countryFilters = ['GB']

# Category filter (subset of or None):
# ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals',
#  'Sports', 'Travel & Events', 'Gaming', 'People & Blogs', 'Comedy (UGC)',
#  'Entertainment', 'News & Politics', 'Howto & Style', 'Education',
#  'Science & Technology', 'Nonprofits & Activism', 'Shows']
#catFilters = None
catFilters = ['Sports', 'Pets & Animals']
#catFilters = ['Sports', 'Gaming']

# Feel free to change the binning.
plot_histogram(master_df, metric, bins=200, countryFilters = countryFilters, catFilters = catFilters)


# # Line Chart
# 
# The Line Chart will show a time-series. It can be plotted according to publish_date or trend_date on the x-axis. The Lineplot can take country or category filters, and the user can control the rollup ('D' for Daily, 'W' for Weekly or 'M' for Monthly) for the x-axis.
# 

# In[20]:


def plot_linechart(df, x_axis:str, y_axis:str, catFilters:list, periodRollup='W', show=True):
    """
    Plots a line chart showing the aggregated trend of a selected metric over time,
    grouped by country and optionally filtered by category.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing YouTube video data, including 'category',
        'country', date fields, and numeric metrics.

    x_axis : str
        The time axis to use for aggregation. Must be one of:
        ['trending_date', 'publish_time']

    y_axis : str
        The metric to aggregate and plot. Must be one of:
        ['views', 'likes', 'dislikes', 'comment_count',
         'log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
         'likes/views', 'dislikes/views', 'comments/views',
         'days_to_trend', 'log_days_to_trend']

    catFilters : list of str
        A list of content categories to include in the plot. Use `df['category'].unique()` to inspect available values.

    periodRollup : str, optional (default='W')
        The resampling frequency to aggregate by. Must be one of the following keys:
        {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}

    show : bool, optional (default=True)
        If True, displays the chart. If False, returns the Plotly figure object for embedding elsewhere.

    Behavior:
    ---------
    - Filters data by category if `catFilters` is provided.
    - Truncates `publish_time` data to after 2017-11-14 to align with reliable trending collection dates.
    - Groups data by country and aggregates the `y_axis` metric using the appropriate function (e.g., mean or sum) based on `expanded_metrics_aggs`.
    - Titles are automatically formatted and include category summaries when applicable.

    Returns:
    --------
    None or plotly.graph_objects.Figure
        Displays the chart if `show=True`; otherwise returns the figure object.
    """
    # Error-check values
    if x_axis not in dateOptions:
        raise ValueError(f"Invalid value {x_axis}. X-axis needs to either be 'trending_date' or 'publish_time'!")

    if y_axis not in expanded_metrics_aggs.keys():
        raise ValueError(f"Invalid value {y_axis}. Y-axis needs to be one of the following values: {expanded_valid_metrics.keys()}")

    if periodRollup not in valid_time_rollups.values():
        raise ValueError(f"Invalid value {periodRollup}. Must be one of the following values: {valid_time_rollups.values()}")

    fig = go.Figure()
    fig.update_layout(height=400)  # set height to be consistent

    title = f'Line Chart: {y_axis} according to {x_axis}: {periodRollup}'
    # Filter by the category list if present
    if catFilters is not None:
        cat_subset = df[df['category'].isin(catFilters)]
        cat_list = ', '.join(catFilters)
        if len(catFilters) > 4:
            cat_list = ', '.join(catFilters[:4]) + f', +{len(catFilters) - 4} more'
        title += f'<br><sup>Categories: {cat_list}</sup>'
    else:
        cat_subset = df.copy()
        title = title + f'<br><sup>No category filters</sup>'

    if x_axis == 'publish_time':
        # The earliest reliable trend_date collection time is '2017-11-14
        # So we should truncate the chart to that
        cat_subset = cat_subset[cat_subset['publish_time'] >= '2017-11-14']

    for country in cat_subset['country'].unique():
        subset_df = cat_subset[cat_subset['country'] == country]
        periodly = subset_df.groupby(subset_df[x_axis].dt.to_period(periodRollup).dt.to_timestamp()).agg({
            y_axis: expanded_metrics_aggs[y_axis]
        })

        fig.add_trace(go.Scatter(
            x = periodly.index,
            y = periodly[y_axis],
            mode='lines',
            name=country,
            fill=None
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        height=500
    )

    if show:
        fig.show()
    else:
        return fig


# In[21]:


# Let's take a look at the line
# 👇 Feel free to modify the variables below to explore how different inputs affect the trend line

# X-axis (date-based options):
# ['trending_date', 'publish_time']
x_axis = 'publish_time'

# Y-axis metric (valid options):
# ['views', 'likes', 'dislikes', 'comment_count',
#  'log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
#  'likes/views', 'dislikes/views', 'comments/views',
#  'days_to_trend', 'log_days_to_trend']
metric = 'log_days_to_trend'

# Category filter (subset of or None):
# ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals',
#  'Sports', 'Travel & Events', 'Gaming', 'People & Blogs', 'Comedy (UGC)',
#  'Entertainment', 'News & Politics', 'Howto & Style', 'Education',
#  'Science & Technology', 'Nonprofits & Activism', 'Shows']
catFilters = ['Music']

# Time rollup (aggregation granularity). Please use the key (not the value):
# {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
periodRollup = 'D'
#catFilters = ['Sports', 'Pets & Animals']
#catFilters = ['Sports', 'Gaming']
plot_linechart(master_df, x_axis, metric, catFilters = catFilters, periodRollup = periodRollup)


# # Scatterplot
# The scatterplot can take in any two metrics for the X- and Y-axes and plot them. We should limit it to the log values of the metrics only (not the absolute numbers). We will color them by country and allow you to hover over individual dots to show the associated video and channel. To speed up execution, we are sampling the data to 300 datapoints.

# In[22]:


def plot_scatterplot(df, x_axis:str, y_axis:str, catFilters:list, show=True):
    """
    Plots a scatterplot comparing two selected metrics across videos, 
    colored by country and optionally filtered by category.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing YouTube video data with 'country', 'category',
        numeric metrics, and text metadata like 'title' and 'channel_title'.

    x_axis : str
        Metric to use on the x-axis. Must be one of:
        ['log_views','log_likes','log_dislikes','log_comment_count',
         'likes/views','dislikes/views','comments/views',
         'days_to_trend', 'log_days_to_trend']

    y_axis : str
        Metric to use on the y-axis. Must follow the same valid options as `x_axis`.

    catFilters : list of str
        Optional list of content categories to include. If None, all categories are shown.
        Use `df['category'].unique()` to see available options.

    show : bool, optional (default=True)
        If True, displays the plot inline. If False, returns the Plotly figure object.

    Behavior:
    ---------
    - Automatically samples up to 3,000 rows per country to improve performance.
    - Distinguishes countries using marker symbols and color.
    - Displays video title and channel name on hover via `customdata`.
    - Legend filtering is enabled via `legendgroup`.

    Returns:
    --------
    None or plotly.graph_objects.Figure
        Displays the plot if `show=True`; otherwise returns the figure object.
    """
    fig = go.Figure()
    fig.update_layout(height=400)  # set height to be consistent

    # Error-check values
    if x_axis not in valid_metrics:
        raise ValueError(f"Invalid value {x_axis}. X-axis needs to be one of {valid_metrics}")

    if y_axis not in valid_metrics:
        raise ValueError(f"Invalid value {y_axis}. Y-axis needs to be one of {valid_metrics}")

    title = f"Scatterplot of {x_axis} by {y_axis}"
    if catFilters is not None:
        cat_subset = df[df['category'].isin(catFilters)]
        cat_list = ', '.join(catFilters)
        if len(catFilters) > 4:
            cat_list = ', '.join(catFilters[:4]) + f', +{len(catFilters) - 4} more'
        title += f'<br><sup>Categories: {cat_list}</sup>'
    else:
        cat_subset = df.copy()
        title = title + f'<br><sup>No category filters</sup>'

    for country in cat_subset['country'].unique():
        subset_df = cat_subset[cat_subset['country'] == country]
        sampled_df = subset_df.sample(n=min(len(subset_df), 3000), random_state=42)  # Speed up rendering by sampling

        fig.add_trace(go.Scattergl(
            x=sampled_df[x_axis],
            y=sampled_df[y_axis],
            mode='markers',
            name=str(country),
            legendgroup=country,   # group legends allow for interactive filtering!
            marker=dict(size=4, opacity=0.25, symbol=country_symbol_map[country]),
            customdata=sampled_df[['title', 'channel_title']],
            hovertemplate=(
                f'<b>Country:</b> {country}' +
                '<br>x: %{x}' +
                '<br>y: %{y}' +
                '<br>Title: %{customdata[0]}' +
                '<br>Channel: %{customdata[1]}' +
                '<extra></extra>'),
            showlegend=True
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        template='plotly_white',
        legend_title='Country'
    )

    if show:
        fig.show()
    else:
        return fig


# In[23]:


# Let's take a look at the line
# 👇 Try different values below to explore relationships between metrics and categories

# X-axis metric (valid options):
# ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
#  'likes/views', 'dislikes/views', 'comments/views',
#  'days_to_trend', 'log_days_to_trend']
x_axis = 'log_dislikes'

# Y-axis metric (same valid options as above)
y_axis = 'log_views'
#catFilters=None

# Category filter (subset of or None):
# ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals',
#  'Sports', 'Travel & Events', 'Gaming', 'People & Blogs', 'Comedy (UGC)',
#  'Entertainment', 'News & Politics', 'Howto & Style', 'Education',
#  'Science & Technology', 'Nonprofits & Activism', 'Shows']
catFilters = ['Music']
#catFilters = ['Sports', 'Pets & Animals']
#catFilters = ['Sports', 'Gaming']
plot_scatterplot(master_df, x_axis, y_axis, catFilters = catFilters)


# # The Static Dashboard and Widgetry
# 
# Okay, let's create our dashboard and control panel now. 
# For the dashboard, we'll keep the histogram and boxplot on top (simpler plots)
# and then the scatter and line chart in the bottom (more complex plots)
# 
# So overall, I think we need the following widgets, starting with the universal
# Universal:
# * y_axis metric [dropdown]
# * catFilters [checkbox]
# 
# Plot Specific:
# * country filter (for histogram) [dropdown]
# * bins (for histogram) [BoundedIntText]
# * x-axis metric (for scatter) [dropdown]
# * x_axis date (for line) [Date picker]
# * periodRollup (for line) [radio button]
# 
# For simplicity, I will eliminate the more relaxed y-axis metric requirements for the line chart.
# 
# For reference, here are the function specs for each
# 1) **Histogram**: def plot_histogram(df, y_axis, bins=50, countryFilters = None, catFilters=None, show=True)
# 2) **ViolinPlot**: def plot_violinplot(df, y_axis, catFilters=None, show=True)
# 3) **Scatter**: def plot_scatterplot(df, x_axis:str, y_axis:str, catFilters:list, show=True)
# 4) **Line**: def plot_linechart(df, x_axis:str, y_axis:str, catFilters:list, periodRollup='W', show=True):
# 
# Let's take the first step and create the static dashboard

# In[30]:


def create_dashboard(df, metric='log_views', catFilters=None, countryFilters=None, 
                     bins=50, xAxisMetric='log_likes', xAxisDate='publish_time', 
                     periodRollup='W'):
    """
    Constructs a 2x2 interactive dashboard using Panel and Plotly to visualize YouTube trending data.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing preprocessed YouTube video data.

    metric : str, optional (default='log_views')
        The metric used for the y-axis across most plots. Must be one of the `valid_metrics`.

    catFilters : list of str, optional
        List of video categories to include. If None, all categories are shown.

    countryFilters : list of str, optional
        List of country codes to filter the dataset for the histogram. 
        Valid values: ['US', 'CA', 'GB']

    bins : int, optional (default=50)
        Number of bins to use in the histogram.

    xAxisMetric : str, optional (default='log_likes')
        Metric to use as the x-axis for the scatterplot. Must be one of the `valid_metrics`.

    xAxisDate : str, optional (default='publish_time')
        Date field used for the x-axis of the line chart.
        Valid options: ['publish_time', 'trending_date']

    periodRollup : str, optional (default='W')
        Time frequency for aggregating the line chart.
        Valid values: 'D' (daily), 'W' (weekly), 'M' (monthly)

    Returns:
    --------
    panel.layout.base.Column
        A Panel layout object containing a titled dashboard with a 2x2 grid of visualizations:
        - Top-left: Histogram
        - Top-right: Violin plot
        - Bottom-left: Scatterplot
        - Bottom-right: Line chart

    Notes:
    ------
    - Uses `plotly` for interactive charts and `panel` for layout management.
    - Plots are wrapped in `pn.pane.Plotly` and arranged using `pn.GridSpec`.
    - Intended for display in Jupyter notebooks or as a standalone Panel app.
    """


    # Let's enable Panel for use in a dashboard
    pn.extension('plotly')

    # 2nd, let's retrieve all the Plotly charts based on the arguments passed
    fig_histogram = plot_histogram(df=df, y_axis=metric, bins=bins, 
	countryFilters=countryFilters, catFilters=catFilters, show=False) 
    fig_violin = plot_violinplot(df=df, y_axis=metric, catFilters=catFilters, show=False)
    fig_scatter = plot_scatterplot(df=df, x_axis=xAxisMetric, y_axis=metric, 
	catFilters=catFilters, show=False)
    fig_line = plot_linechart(df=df, x_axis=xAxisDate, y_axis=metric, 
	catFilters=catFilters, periodRollup=periodRollup, show=False)

    # 3rd, we wrap each figure in a Panel pane
    histogram_panel = pn.pane.Plotly(fig_histogram, sizing_mode="stretch_width", height=400)
    violin_panel = pn.pane.Plotly(fig_violin, sizing_mode="stretch_width", height=400)
    scatter_panel = pn.pane.Plotly(fig_scatter, sizing_mode="stretch_width", height=400)
    line_panel = pn.pane.Plotly(fig_line, sizing_mode="stretch_width", height=400)

    # Organize all the panels in a 2x2 grid
    grid = pn.GridSpec(sizing_mode="stretch_width", max_height = 800)
    grid[0, 0] = histogram_panel
    grid[0, 1] = violin_panel
    grid[1, 0] = scatter_panel
    grid[1, 1] = line_panel

    dashboard = pn.Column("Trending YouTube Dataset Dashboard", grid)

    return dashboard
    # Web display



# Let's have a look at what this static dashboard will look like!

# In[31]:


dashboard = create_dashboard(master_df)
dashboard
#!python3 -m panel serve assignment3.ipynb --port 5006 --allow-websocket-origin=*


# ## Widgets
# 
# We now define all the widgets that will power our interactive dashboard. We're using Panel because it allows us to export the dashboard as a self-contained .html file—perfect for viewing locally without needing a running server.
# 
# These widgets let us control which data is displayed in each of the four charts: histogram, violinplot, scatterplot, and line chart.
# 
# Before diving into the widgets, here’s a recap of the key variables and value lists we’ve defined earlier:
# 
# 1. **Available Countries (for histogram and line plot filtering)**: Each country is associated with a shape (used in scatterplots).
# 
# ```country_symbol_map = {'US': 'circle', 'CA': 'square', 'GB': 'diamond'}```
# 
# 
# 2) **Valid Metrics (for y-axis on all charts and x-axis for scatterplots)**: These include both raw and transformed versions of standard YouTube stats.
# 
# ```valid_metrics = ['log_views','log_likes','log_dislikes','log_comments','likes/views','dislikes/views','comments/views','days_to_trend', 'log_days_to_trend']```
# 
# 3) **Date Fields (used for the x-axis of line charts)**: Choose between when a video was published or when it trended.
# ```dateOptions = ['trending_date', 'publish_time']```
# 
# 4) **Time Rollups (used to control aggregation in line charts)**: Controls the grouping frequency for time series data.
# ```valid_time_rollups = {'Daily':'D', 'Weekly: 'W', 'Monthly': 'M'}```
# 
# Now let’s define our widget controls:
# 
# * **Dropdowns** to choose metrics and date fields
# * **Checkbox groups** to filter categories and countries
# * **Slider** for adjusting histogram bin count
# * **Radio buttons** to control rollup granularity
# * **A Submit button** to trigger updates (avoids auto-refresh when toggling multiple filters)
# 
# We’ll organize them into two columns:
# * The **left column** holds dashboard-wide settings.
# * The **right column** contains chart-specific controls.
# 
# Go ahead and experiment by changing any of these values. You'll see how they influence the charts once we render the dashboard.
# 

# In[32]:


# The widget layour looks pretty awesome! Now, let's create the actual interactive dashboard!
# 
# # Interactive Dashboard
# 
# Assembling the interactive dashboard should be pretty straightforward, now that we have the function for the static dashboard and all the available widgets for the control panel in place.

# In[33]:


#@pn.depends(yAxisDropdown, categoryCheckbox,countryCheckbox,binsField,
#            xAxisScatterDropdown,xAxisLineDropdown,PeriodRollupRadio)
def create_interactive_dashboard(metric, catFilters, countryFilters, 
                     bins, xAxisMetric, xAxisDate, periodRollup, controlPanel):
    """
    Constructs and returns an interactive Panel dashboard based on selected widget values.

    Parameters:
    -----------
    metric : str
        The Y-axis metric to display across all charts (e.g., 'log_views', 'likes/views').
    catFilters : list or None
        A list of selected video categories to filter the dataset (e.g., ['Music', 'Gaming']).
    countryFilters : list or None
        A list of selected countries to filter the dataset (e.g., ['US', 'GB']).
    bins : int
        Number of bins to use in the histogram plot.
    xAxisMetric : str
        The X-axis metric used for the scatterplot (must be one of the valid_metrics).
    xAxisDate : str
        Date field for line plot X-axis. Must be either 'publish_time' or 'trending_date'.
    periodRollup : str
        Time aggregation level for the line chart (e.g., 'Daily', 'Weekly', 'Monthly').

    Returns:
    --------
    pn.Column
        A Panel Column object combining the control panel and the four chart panels (histogram,
        violinplot, scatterplot, line chart), rendered based on user-specified settings.
    """
    print(f"Bindings. metric={metric}, catFilters={catFilters}, countryFilters={countryFilters}, bins={bins}")
    print(f"xAxisMetric={xAxisMetric}, xAxisDate={xAxisDate}, periodRollup={periodRollup}")
    dashboard = create_dashboard(df=master_df, 
                                 metric=metric, 
                                 catFilters=catFilters, 
                                 countryFilters=countryFilters,
                                 bins=bins, 
                                 xAxisMetric=xAxisMetric, 
                                 xAxisDate=xAxisDate, 
                                 periodRollup=valid_time_rollups[periodRollup])


    return pn.Column(
        controlPanel,
        pn.Spacer(height=20),
        dashboard,
        sizing_mode='stretch_both')


# Now that we’ve defined all our widgets and charts, it’s time to bring the interactive dashboard to life.
# 
# We provide two modes of dashboard interaction:
# 
# 1. **Reactive Mode (commented out)**
# This mode uses pn.bind(...) to automatically update the dashboard whenever any widget is changed. While simple, it can be slow and frustrating—especially when toggling many checkboxes (like categories).
# 
# 2. **Submit-Based Mode (active)**
# This setup disables auto-refresh and requires users to click the Submit button to apply changes. It offers better performance and avoids unnecessary re-renders when experimenting with filters.
# 
# By default, we initialize the dashboard once with the default widget values. You can:
# * Edit the defaults in the widget cell above,
# * Re-run the Submit button cell to update the output, or
# * Uncomment the reactive mode if you want live updates instead.
# 
# 

# In[34]:


#dashboard_panel = pn.panel(create_interactive_dashboard)
#dashboard_panel.servable()  # Optional in notebooks

# Bind function to widgets directly if we want automated updates. But it is rather slow
# Uncomment the following code to see the reactive version.
#dashboard_panel = pn.bind(create_interactive_dashboard,
#                          yAxisDropdown, categoryCheckbox, countryCheckbox, binsField,
#                          xAxisScatterDropdown, xAxisLineDropdown, PeriodRollupRadio)
#dashboard_panel

# Defining an onSubmit event because it is a pain deselecting category buttons. 
# Comment this code out and uncomment the code above to switch from event-based mode to reactive mdoe (which is kinda slow)
def get_interactive_dashboard_app():
	pn.extension('plotly')
	
	# Widget definitions
	

	# define the y_axis metric [dropdown]
	yAxisDropdown = pn.widgets.Select(name="Primary Metric", 
									  options=valid_metrics, 
									  value='log_views')

	#catFilters [checkbox]
	categoryCheckbox = pn.widgets.CheckBoxGroup(name='Categories',
												options=valid_categories,
												value=valid_categories)
	# Because it's a pain if the dashboard keeps on refreshing when you select/deselect checkboxes
	# I'll add a submit button
	submitButton = pn.widgets.Button(name='Submit', button_type='primary')

	#Plot Specific:
	#country filter (for histogram) [checkbox]
	countryCheckbox = pn.widgets.CheckBoxGroup(name="Countries (for histogram)",
											   options=list(country_symbol_map.keys()),
											   value=list(country_symbol_map.keys()))
	#countryCheckbox = pn.widgets.MultiChoice(name="Countries (for histogram)",
	#                                           options=list(country_symbol_map.keys()),
	#                                           value=list(country_symbol_map.keys()))

	# bins (for histogram) [BoundedIntText]
	#binsField = pn.widgets.IntInput(name="Histogram bins",
	#                                value=50,
	#                                start=10,
	#                                end=300)
	binsField = pn.widgets.IntSlider(name="Histogram bins",
									 start=10,
									 end=300,
									 value=50)

	# x-axis metric (for scatter) [dropdown]
	xAxisScatterDropdown = pn.widgets.Select(name="X-axis metric for scatterplot",
									  options = valid_metrics,
									  value='log_likes')


	# x_axis date (for line) [dropdown]
	xAxisLineDropdown = pn.widgets.Select(name="X-axis date type for lineplot",
										  options = dateOptions,
										  value='trending_date')

	# periodRollup (for line) [radio button]
	#PeriodRollupRadio = pn.widgets.RadioButtonGroup(name="Granularity for lineplot",
	#                                                options=list(valid_time_rollups.keys()),
	#                                                button_type='primary')
	PeriodRollupRadio = pn.widgets.RadioBoxGroup(
		name='Granularity for lineplot',
		options=list(valid_time_rollups.keys()),
		value='Weekly'
	)
	# [2] Output container
	dashboard_container = pn.Column()

	# [3] Control panel
	controlPanel = pn.Row(
		pn.Column(
			pn.pane.Markdown("###Controls for entire dashboard"),
			yAxisDropdown,
			pn.pane.Markdown("---- Category Selection ----"),
			categoryCheckbox,
			sizing_mode="stretch_width"
		),
		pn.Column(
			pn.pane.Markdown("###Chart-specific controls"),
			pn.pane.Markdown("---- Country Selection for Histogram ----"),
			countryCheckbox,
			binsField,
			xAxisScatterDropdown,
			xAxisLineDropdown,
			pn.pane.Markdown("---- Rollup for Lineplot ----"),
			PeriodRollupRadio,
			submitButton,
			sizing_mode="stretch_width"
		)
	)
	def on_submit(event):
		dashboard = create_interactive_dashboard(
			metric=yAxisDropdown.value,
			catFilters=categoryCheckbox.value,
			countryFilters=countryCheckbox.value,
			bins=binsField.value,
			xAxisMetric=xAxisScatterDropdown.value,
			xAxisDate=xAxisLineDropdown.value,
			periodRollup=PeriodRollupRadio.value,
			controlPanel=controlPanel
		)
		dashboard_container.objects = [pn.Spacer(height=20), dashboard]

	submitButton.on_click(on_submit)
	on_submit(None)

	return pn.Column(controlPanel, dashboard_container)

# # Success!
# So we got the interactive dashboard working within Jupyter. Huzzah!There are a l
# There are a lot of stories that can be discovered from this dashboard. Let's take a closer look at a specific cut of the dashboard:
# 
# **Control Panel Settings**
# ![image.png](attachment:306f4f37-f63b-4b36-afd4-bfb316633559.png)
# 
# **Dashboard** 
# ![image.png](attachment:feac6b83-77f7-4e84-aa2a-30e3fab0fd0d.png)
# 
# This dashboard provides a deep dive into how log-transformed dislikes vary across countries and time for the News & Politics category, using data from the United States, Canada, and Great Britain.
# 
# 🔧 Control Settings:
# * **Primary Metric (Y-axis)**: log_dislikes
# * **Categories Filtered**: News & Politics only
# * **Countries Filtered (for Histogram)**: US, CA, GB
# * **Histogram Bins**: 50
# * **Scatter X-axis Metric**: log_likes
# * **Line Chart X-axis Date**: trending_date
# * **Line Chart Time Rollup**: Daily
# 
# ### Analysis
# * In the **Histogram** you can see that the ***log dislikes*** are roughly bell-shaped and normally distributed, with a common range of hundreds to thousands of dislikes per video. The long tail suggests a few highly disliked videos
# * In the **ViolinPlot** the US and GB show higher median dislike counts than CA. GB has a far wider violin and suggests greater variability in viewer reaction. The higher interquartile range suggests a stronger polarization in GB than US and Canada!
# * In the **Scatterplot** there is a clear positive correlation between log_likes and log_dislikes, with all three countries showing the same trend. High engagement on both ends suggest polarization all around
# * Finally, in the **line chart** suggests an upward trend in dislikes, during the Q1 and Q2 2018 time period for the US. CA is mostly flat -- and they tend to dislike videos at a lower rate, while GB is very erratic and presides over a higher baseline of log_dislikes than either country. Who knew the Brits were so apt to disliking YT videos!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]: