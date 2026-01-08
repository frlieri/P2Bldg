"""
src/export/analysis_plots.py

Plotting utilities to visualize time-series, typical-day analyses and diagnostics
for building and battery optimisation workflows. The module provides convenience
wrappers around Plotly (primary), with some Matplotlib/Seaborn helpers for
statistical plots.

Main responsibilities:

- Flexible time-series plotting (lines, stacked areas, secondary y-axis) with
  `plot_timeseries`.
- Composition of multiple Plotly figures into a subplot grid while avoiding
  duplicate legend entries via `plot_subplots`.
- Typical-day quantile calculation and plotting helpers:
  `calc_dayprofile_quantiles`, `plot_quantiles`, `plot_typical_day_quantiles`,
  and `compare_typical_day_quantiles`.
- Scatterplots with linear regression overlays using `scatterplot`.
- Domain-specific visualisations such as `plot_heat_curve` (temperatures, COP)
  and `plot_seasonal_boxplots` (hourly boxplots by season).

Key behaviours and expectations:

- Input data: pandas `DataFrame` or `Series` with a datetime-like index.
- Most functions return a Plotly `go.Figure` and optionally call `.show()` by
  default (display side-effect). They do not perform file I/O.
- Several functions accept plotting options to control styles, stacking and
  secondary axes; quantile helpers expect hourly/resampled data for typical-day
  aggregation.

Dependencies:

- pandas, numpy, plotly, seaborn, matplotlib and project constants from
  `src.const` (e.g. `NOM_TEMP_INDOOR`, `TimeseriesCols`).

Intended use:

- Interactive exploration, dashboarding and figure generation for analysis and
  reporting. Functions are lightweight wrappers aimed at consistent, reusable
  visual output for downstream notebooks and reporting scripts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from src.const import NOM_TEMP_INDOOR, TimeseriesCols


def plot_timeseries(df: pd.DataFrame|pd.Series, title='', plot_options=None, show=True, y_range: list=None):
    """
    Plots time series data from a DataFrame using Plotly.

    Parameters:
    df (pd.DataFrame): DataFrame with time series data. Index should be the time index.
    plot_options (dict): Optional dictionary to specify plot types, linestyles, and colors for each column.
                         Example:
                         {
                             'column1': {'type': 'line', 'line_style': 'dash', 'color': 'blue'},
                             'column2': {'type': 'area', 'color': 'red', 'fill': 'tonexty'}
                         }

    Returns:
    fig (go.Figure): Plotly Figure object.
    """

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    fig = go.Figure()

    for col in df.columns:
        # Default plot options
        plot_type = 'line'
        line_style = 'solid'
        fill = None
        color = None
        stackgroup = None
        pattern_shape = None
        secondary_y = False  # Default to primary y-axis

        if plot_options and col in plot_options:
            col_options = plot_options[col]
            plot_type = col_options.get('type', 'line')
            line_style = col_options.get('line_style', 'solid')
            fill = col_options.get('fill', None)
            color = col_options.get('color', None)
            stackgroup = col_options.get('stackgroup', None)
            pattern_shape = col_options.get('pattern_shape', None)
            secondary_y = col_options.get('secondary_y', False)

        if plot_type == 'line':
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode='lines',
                line=dict(dash=line_style, color=color),
                name=col,
                yaxis="y2" if secondary_y else "y1"
            ))
        elif plot_type == 'area':
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], fill=fill,
                mode='lines', line=dict(color=color),
                name=col,
                stackgroup=stackgroup,
                fillpattern=dict(shape=pattern_shape) if pattern_shape else None,
                yaxis="y2" if secondary_y else "y1"
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Hours',
        yaxis_title='P in kW',
        yaxis=dict(
            range=y_range
        ),
        yaxis2=dict(
            title="EUR/kWh_el or kgCO2eq./kWh_el",
            overlaying='y',  # Overlay the secondary y-axis on the same plot
            side='right'
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=df.index[::24],
        ),
    )

    if show:
        fig.show()

    return fig


def plot_subplots(figures, rows, cols, y_ranges=None, show=True):
    """
    Arranges multiple Plotly figures as subplots with individual titles and unique legend items.

    Parameters:
    figures (list): List of Plotly Figure objects.
    rows (int): Number of rows in the subplot layout.
    cols (int): Number of columns in the subplot layout.
    y_ranges (list): List of tuples specifying (ymin, ymax) for each subplot.

    Returns:
    fig (go.Figure): Combined Plotly Figure object with subplots.
    """
    if len(figures) != rows * cols:
        raise ValueError("The number of figures must match the number of subplots (rows * cols).")

    if y_ranges is not None and len(y_ranges) != len(figures):
        raise ValueError("The number of y_ranges must match the number of figures.")

    # Extract titles from the individual figures
    titles = [figure.layout.title.text for figure in figures]

    # Create a subplot figure
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    # Track unique legend items
    unique_legends = set()

    # Iterate through figures and add them to the subplot figure
    for i, individual_fig in enumerate(figures):
        row = (i // cols) + 1
        col = (i % cols) + 1

        for trace in individual_fig.data:
            # Add trace name to the unique legend set
            if trace.name not in unique_legends:
                unique_legends.add(trace.name)
                show_legend = True
            else:
                show_legend = False

            # Add the trace to the subplot
            fig.add_trace(trace.update(showlegend=show_legend), row=row, col=col)

        # Set y-range for each subplot if provided
        if y_ranges:
            ymin, ymax = y_ranges[i]
            fig.update_yaxes(range=[ymin, ymax], row=row, col=col)

        # Find the correct x-axis and y-axis references for this subplot
        xaxis_ref = f'xaxis{(i + 1) if i > 0 else ""}'  # xaxis1, xaxis2, etc.
        yaxis_ref = f'yaxis{(i + 1) if i > 0 else ""}'  # yaxis1, yaxis2, etc.

        # Update the X-axis for the subplot
        fig.update_layout({
            xaxis_ref: dict(
                title_text=individual_fig.layout.xaxis.title.text,  # Copy title
                tickmode=individual_fig.layout.xaxis.tickmode,  # Copy tickmode
                tickvals=individual_fig.layout.xaxis.tickvals,  # Copy tickvals
            ),
            yaxis_ref: dict(
                title_text=individual_fig.layout.yaxis.title.text,  # Copy title
                range=y_ranges[i],
            )
        })

        # Update the secondary Y-axis (right side, overlaying primary Y-axis) if it exists
        if 'yaxis2' in individual_fig.layout:
            fig.update_layout({
                yaxis_ref + '2': dict(
                    title_text=individual_fig.layout.yaxis2.title.text,  # Copy title for secondary axis
                    overlaying="y",  # Overlay on primary Y-axis
                    side="right",  # Display on right side
                    range=(-1,1),
                )
            })

    fig.update_layout(title='')

    if show:
        fig.show()

    return fig


def plot_typical_day_quantiles(ts: pd.Series, quantiles=None, title='Typedayplot', yaxis_title='',
                               fig=None, row=None, col=None, yaxis_range=None):
    """
    Plots a typical day quantile analysis of the provided time series data. This function calculates
    the quantiles for hourly data based on the input time series, creates a plot using the specified
    or default quantiles, and visualizes the results with appropriate layout and titles. It also
    allows customization of the figure, titles, and y-axis range based on provided inputs.

    :param ts: Series containing time-series data to be analyzed.
    :type ts: pandas.Series
    :param quantiles: List of desired quantile values to compute and plot. Defaults to None.
    :type quantiles: list, optional
    :param title: Title of the plot. Defaults to 'Typedayplot'.
    :type title: str, optional
    :param yaxis_title: Label for the y-axis of the plot. Defaults to an empty string.
    :type yaxis_title: str, optional
    :param fig: Existing figure to use for the plot. If None, a new figure is created. Defaults to None.
    :type fig: plotly.graph_objs._figure.Figure, optional
    :param row: Row index for locating the plot on a subplot. Applicable only if a figure is provided.
                 Defaults to None.
    :type row: int, optional
    :param col: Column index for locating the plot on a subplot. Applicable only if a figure is provided.
                 Defaults to None.
    :type col: int, optional
    :param yaxis_range: Range for the y-axis in the plot. If None, the range is determined automatically.
                         Defaults to None.
    :type yaxis_range: tuple, optional
    :return: None
    """

    quan_df = calc_dayprofile_quantiles(ts, quantiles)

    fig = make_subplots(rows=1, cols=1)

    fig = plot_quantiles(quan_df, fig=fig, row=1, col=1, yaxis_range=yaxis_range)

    # Set layout
    fig.update_layout(title=title,
                      xaxis_title='Hour of the Day',
                      yaxis_title=yaxis_title)

    fig.show()


def compare_typical_day_quantiles(ts_list: [pd.Series], quantiles=None, subplot_titles=None, title='Typedayplots',
                                  yaxis_title='', fig=None, row=None, col=None, yaxis_range=None):
    """
    Generates a comparative visualization of quantiles for multiple time series datasets.

    This function compares the typical day's quantiles for multiple time series datasets
    by plotting individual subplots for each dataset and its respective quantiles. The
    layout of the figure and other aspects such as titles, axis labels, and ranges can
    be customized.

    :param ts_list: A list of pandas Series objects representing the time series data to
        compare.
    :param quantiles: A list of desired quantiles for comparison. If None, default quantiles
        are used.
    :param subplot_titles: A list of titles for the individual subplots. Its length must
        match the number of datasets in `ts_list`.
    :param title: The main title for the entire figure. Default is 'Typedayplots'.
    :param yaxis_title: The title of the y-axis for the entire figure. Default is an empty
        string.
    :param fig: An existing figure object to plot onto. If None, a new figure is created.
    :param row: The row position in the grid layout of the figure where the plots will
        be drawn. Default is None.
    :param col: The column position in the grid layout of the figure where the plots will
        be drawn. Default is None.
    :param yaxis_range: Range for the y-axis, specified as a list [min, max]. If None,
        it is automatically calculated.
    :return: Returns None. Displays the generated plot.
    """

    if subplot_titles is not None:
        assert len(subplot_titles) == len(ts_list), "number of datasets and subplot titles doesn't match"

    fig = make_subplots(rows=1, cols=len(ts_list), subplot_titles=subplot_titles)

    for i, ts in enumerate(ts_list):
        quan_df = calc_dayprofile_quantiles(ts, quantiles)
        fig = plot_quantiles(quan_df, fig=fig, row=1, col=i+1, yaxis_range=yaxis_range)

    # Set layout
    fig.update_layout(title=title,
                      xaxis_title='Hour of the Day',
                      yaxis_title=yaxis_title)

    if yaxis_range is None:
        min_val, max_val = min([ts.min() for ts in ts_list]), max(ts.max() for ts in ts_list)
        yaxis_range = [min_val-0.1*abs(min_val), max_val+0.1*abs(max_val)]
        fig.update_yaxes(range=yaxis_range)

    fig.show()


def calc_dayprofile_quantiles(ts: pd.Series, quantiles=None):
    if quantiles is None:
        quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    assert len(quantiles) % 2 != 0 and [q for q in quantiles if q == 0.5], \
        "Invalid quantiles gives. Length of list has to be odd number and list has to contain 0.5 for median"

    # Calculate quantiles
    return ts.groupby(ts.index.time).quantile(quantiles).unstack(level=1)


def plot_quantiles(quan_df: pd.DataFrame, fig=None, row=None, col=None, yaxis_range=None):
    """
    Plots quantile ranges and median from a given DataFrame on a figure using Plotly.

    This function creates a visual representation of quantiles from a DataFrame, allowing you
    to fill areas between quantile ranges and overlay a median line. It supports both standalone
    figures and subplots with specified rows and columns.

    :param quan_df: A pandas DataFrame where the index represents the time component for the x-axis
        and the columns represent the quantiles to plot.
    :param fig: An optional Plotly Figure object. Defaults to None, in which case a new figure is
        created.
    :param row: Integer representing the row number in the subplot layout where the plot will
        be added (optional, but required if `fig` is provided).
    :param col: Integer representing the column number in the subplot layout where the plot will
        be added (optional, but required if `fig` is provided).
    :param yaxis_range: An optional parameter for the y-axis range. This is not utilized in the
        function as provided.
    :return: A Plotly Figure object containing the plotted quantiles and median line.
    """

    # Extract time component for x-axis
    ind = quan_df.index.to_list()

    # Create figure
    if fig is None:
        fig = go.Figure()
    else:
        assert row and col, "row and col must be given for subplot"

    # Fill areas between quantile ranges
    for i, quan in enumerate(quan_df.columns):
        if quan == 0.5:
            break
        quan_range = (quan, quan_df.columns[-(i+1)])
        opacity = 0.25 + i/(len(quan_df.columns) / 2 - 1)*0.75
        fig.add_trace(go.Scatter(x=ind + ind[::-1],
                                 y=(quan_df.loc[:, quan_range[0]].to_list() +
                                    quan_df.loc[:, quan_range[1]].to_list()[::-1]),
                                 fill='toself',
                                 fillcolor=f'rgba(128, 128, 128, {opacity})',  # color code + opacity
                                 line=dict(color='rgba(255,255,255,0)'),  # not visible
                                 name=f'{quan_range[0]}-{quan_range[1]} Quantile Range'),
                      row=row, col=col)



    # Add median line
    fig.add_trace(go.Scatter(x=ind, y=quan_df.loc[:, 0.5], mode='lines', name='Median',
                  line=dict(color='rgba(0,0,0,1.0)')), row=row, col=col)

    # Show plot
    return fig


def scatterplot(data, x=None, y1=None, y2=None, graph_title=None, x_axis_title=None, y1_axis_title=None,
                y2_axis_title=None):
    """
    Create and show a scatter plot for a DataFrame with optional dual y-axes and linear regression lines.

    Parameters:
    - data (pd.DataFrame): The data for the scatter plot.
    - x (str, optional): The column name for the x-axis. Defaults to DataFrame index.
    - y1 (list of str, optional): List of column names for the first y-axis. Defaults to all columns if y2 is not provided.
    - y2 (list of str, optional): List of column names for the second y-axis.
    - graph_title (str, optional): The title of the graph.
    - x_axis_title (str, optional): The title of the x-axis.
    - y1_axis_title (str, optional): The title of the first y-axis.
    - y2_axis_title (str, optional): The title of the second y-axis.

    Returns:
    - fig: A Plotly figure object.
    """

    # Set default x and y1 values if not provided
    if x is None:
        data = data.reset_index()
        x = data.columns[0]
    if y1 is None:
        y1 = data.columns.tolist()
        y1.remove(x)
        if y2:
            for col in y2:
                if col in y1:
                    y1.remove(col)

    # Create a figure with secondary y-axis if y2 is provided
    specs = [[{"secondary_y": True}]] if y2 else [[{"secondary_y": False}]]
    fig = make_subplots(specs=specs)

    # Add traces and regression lines for the first y-axis
    for col in y1:
        fig.add_trace(
            go.Scatter(x=data[x], y=data[col], mode='markers', name=col),
            secondary_y=False,
        )
        # Calculate and add the linear regression line
        m, b = np.polyfit(data[x], data[col], 1)
        fig.add_trace(
            go.Scatter(x=data[x], y=m * data[x] + b, mode='lines', name=f'{col} Trend', line=dict(dash='dash')),
            secondary_y=False,
        )

    # Add traces and regression lines for the second y-axis if provided
    if y2:
        for col in y2:
            fig.add_trace(
                go.Scatter(x=data[x], y=data[col], mode='markers', name=col),
                secondary_y=True,
            )
            # Calculate and add the linear regression line
            m, b = np.polyfit(data[x], data[col], 1)
            fig.add_trace(
                go.Scatter(x=data[x], y=m * data[x] + b, mode='lines', name=f'{col} Trend', line=dict(dash='dash')),
                secondary_y=True,
            )

    # Set the titles
    if graph_title:
        fig.update_layout(title=graph_title)
    if x_axis_title:
        fig.update_xaxes(title_text=x_axis_title)
    if y1_axis_title:
        fig.update_yaxes(title_text=y1_axis_title, secondary_y=False)
    if y2_axis_title:
        fig.update_yaxes(title_text=y2_axis_title, secondary_y=True)

    fig.show()
    return fig


def plot_heat_curve(df, indoortemp=NOM_TEMP_INDOOR):
    """
    Generates and displays a heat curve plot using the given dataframe and indoor temperature.

    The plot visually represents various temperature-related data over time using a combination of
    primary and secondary y-axes. It includes external temperature, forward flow temperature,
    indoor temperature, and optionally, COP (if present in the dataframe). The chart also highlights
    the heating demand region by shading the area between the external temperature and the specified
    indoor temperature.

    :param df: The input dataframe containing indexed time-series data for 'Aussentemperatur',
               'Vorlauftemperatur', and optional 'COP' values.
               - Index: Time-series index
               - Columns:
                   - `Aussentemperatur`: External temperature data
                   - `Vorlauftemperatur`: Forward flow temperature data
                   - `COP` (optional): Coefficient of performance data
    :param indoortemp: The assumed constant indoor temperature to compare against external
                       temperature and calculate heating demand. Default is set to `NOM_TEMP_INDOOR`.

    :return: None
    """
    # Create a figure with subplots (2 y-axes)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add line plot for Aussentemperatur
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Aussentemperatur'],
        mode='lines',
        name='Aussentemperatur',
        line=dict(color='blue')
    ))

    # Add line plot for Vorlauftemperatur
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Vorlauftemperatur'],
        mode='lines',
        name='Vorlauftemperatur',
        line=dict(color='orange')
    ))

    if 'COP' in df.columns:
        # Add line plot for COP with secondary y-axis
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['COP'],
            mode='lines',
            name='COP',
            line=dict(color='black', dash='dash'),
        ), secondary_y=True)

    # Fill the area between Aussentemperatur and the given indoortemp
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[indoortemp] * len(df),  # Constant line for indoortemp
        mode='lines',
        name='Innentemperatur',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Aussentemperatur'],
        fill='tonexty',  # Fill to the next trace (Aussentemperatur)
        fillcolor='rgba(200, 130, 0, 0.2)',  # Fill color
        mode='none',  # No line
        name='Heizbedarf',
    ))

    # Update layout
    fig.update_layout(
        title='Heizkurve',
        xaxis_title='Stunden',
        yaxis_title='Temperatur (°C)',
        template='plotly',
        yaxis2=dict(
            title='COP',  # Label for the second y-axis
            overlaying='y',  # Allow the two y-axes to share the x-axis
            side='right',  # Position the second y-axis on the right
            range=[0, None],
        )
    )

    # Show the figure
    fig.show()


def plot_seasonal_boxplots(df: pd.DataFrame, col=str):
    """
    Generate seasonal boxplots of a specified column against the hour of the day, categorized by
    season. This function helps visualize data distribution and variation for each hour, grouped by
    seasons (e.g., Winter, Spring, etc.).

    :param df: The input pandas DataFrame containing the time-series data. It must include
        the columns corresponding to seasons and hours of the day for the function to properly
        categorize and plot.
    :param col: The name of the column in the DataFrame to be visualized. This column's values
        will be plotted on the y-axis of the boxplot.
    :return: None. This function renders and displays the boxplot visualization.

    """

    df[TimeseriesCols.Season] = df[TimeseriesCols.Season].astype('category')

    # Set the figure size and grid for the plot
    plt.figure(figsize=(12, 8))

    # Create the boxplot for each season
    sns.boxplot(x=TimeseriesCols.Hour, y=col, hue=TimeseriesCols.Season, data=df, palette='Set2')

    # Set plot titles and labels
    plt.title('Boxplot of Values by Hour of the Day for Each Season')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Values')

    # Show the legend for the seasons
    plt.legend(title='Season')

    # Display the plot
    plt.xticks(range(24))  # Ensure the hours are correctly displayed (0 to 23)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    hourly_data = np.random.normal(loc=50, scale=10, size=len(date_range))
    df = pd.DataFrame(hourly_data, index=date_range, columns=['Value'])
    plot_typical_day_quantiles(df.iloc[:, 0])

    hourly_data1 = np.random.normal(loc=50, scale=10, size=len(date_range))
    hourly_data2 = np.random.normal(loc=55, scale=12, size=len(date_range))
    day1_df = pd.DataFrame(hourly_data1, index=date_range, columns=['Value']).iloc[:, 0]
    day2_df = pd.DataFrame(hourly_data2, index=date_range, columns=['Value']).iloc[:, 0]
    compare_typical_day_quantiles([day1_df, day2_df])