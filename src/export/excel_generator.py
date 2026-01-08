"""
src/export/excel_generator.py

Utilities to export pandas DataFrames to an Excel workbook and to insert
XlsxWriter charts based on declarative graph specifications.

Responsibilities:

- write multiple DataFrames to separate worksheets in a single `.xlsx` file.
- create and insert XlsxWriter charts configured via a compact `graph_specs`
  dictionary, including support for primary/secondary charts, dual y-axis,
  trendlines, custom series styles and selective legend hiding.
- handle common pandas structures (flat columns, MultiIndex columns, MultiIndex
  index) so that chart ranges and category columns are written correctly.

Public functions:

- add_chart_to_worksheet(workbook, worksheet, graph_nr, graph_specs, df, dfname)
    Build and add one (possibly combined) XlsxWriter chart to the given worksheet
    using the provided DataFrame and `graph_specs`.

- save_xlsx_wb(out_path, df_dict, graphs=None)
    Write a workbook with one sheet per DataFrame in `df_dict` and insert charts
    described in `graphs`. Uses the `xlsxwriter` engine via `pandas.ExcelWriter`.

Key behavior and expectations:

- Requires `pandas` and the `xlsxwriter` engine available to `pandas.ExcelWriter`.
- DataFrame indexes are written to Excel; index names are cleared to avoid extra
  header cells. MultiIndex columns are flattened for export so column lookups
  used to configure chart series remain stable.
- Chart series use Excel row/column ranges built from the exported sheet layout.
  If a requested column name is missing, the function logs a message and skips it.
- `graph_specs` is declarative and supports keys such as `graph_type_1`,
  `graph_type_2`, `graph_type_2_cols`, `cols_to_include`, `skip_first_n_rows`,
  `skip_last_n_rows`, `position`, `x_y_scale`, `title`, `x_axis_title`,
  `y_axis_title`, `font_size`, `column_styles`, `delete_series_from_legend`,
  and `trendline`.
- Trendline color is taken from the corresponding column style entry; ensure
  `column_styles` contains the expected color path if trendlines are used.

Example (concept):
    graphs = {
        "sheet": [
            {
                "graph_type_1": {"type": "line"},
                "cols_to_include": ["col_a", "col_b"],
                "position": "B3",
                "x_y_scale": [1.5, 1.0],
            }
        ]
    }

Return values:
- Both helpers perform file/worksheet side effects and return `None`.

"""

import pandas as pd


def add_chart_to_worksheet(workbook, worksheet, graph_nr: int, graph_specs: dict, df: pd.DataFrame,
                           dfname: str):
    """
    Adds a chart to the specified worksheet based on the provided specifications and dataframe.

    Summary:
    This function generates Excel charts using the provided workbook, worksheet, graph specifications,
    and data (df). It supports adding multiple charts, customizing their appearance (e.g., titles,
    axis labels, font sizes, etc.), and handles features like trendlines, dual-axis charts, and selective
    legend customization. The charts can be positioned and scaled according to the `graph_specs` provided.

    :param workbook: The Excel workbook where the chart is to be created.
    :param worksheet: The worksheet within the workbook where the chart will be inserted.
    :param graph_nr: An integer indicating the index of the graph for placement on the worksheet.
    :param graph_specs: A dictionary containing configurations for the chart, such as type, column styles,
        axis labels, font sizes, trendlines, and positioning details.
    :param df: The pandas DataFrame containing the data to be plotted.
    :param dfname: The name of the worksheet (string) with the reference data for the chart.
    :return: None

    Minimal example::
    graph_specs =
                    {
                        "graph_type_1": {"type": "line"},
                        "cols_to_include": ["col_a", "col_b"],
                        "position": "B3",
                        "x_y_scale": [2.0, 1.0],
                    }

    List of possible graph options
    {'type': 'area'},
    {'type': 'area', 'subtype': 'stacked'},
    {'type': 'area', 'subtype': 'percent_stacked'},
    {'type': 'bar'},
    {'type': 'bar', 'subtype': 'stacked'},
    {'type': 'bar', 'subtype': 'percent_stacked'},
    {'type': 'column'},
    {'type': 'column', 'subtype': 'stacked'},
    {'type': 'column', 'subtype': 'percent_stacked'},
    {'type': 'line'},
    {'type': 'line', 'subtype': 'stacked'},
    {'type': 'line', 'subtype': 'percent_stacked'},
    {'type': 'pie'},
    {'type': 'pie', 'subtype': 'exploded'},
    {'type': 'doughnut'},
    {'type': 'scatter'},
    {'type': 'scatter', 'subtype': 'straight'},
    {'type': 'scatter', 'subtype': 'straight_with_markers'},
    {'type': 'scatter', 'subtype': 'smooth'},
    {'type': 'scatter', 'subtype': 'smooth_with_markers'},
    {'type': 'radar'},
    {'type': 'radar', 'subtype': 'with_markers'},
    {'type': 'radar', 'subtype': 'filled'}

    dash type options
    'solid': A solid line.
    'dash': A dashed line.
    'dot': A dotted line.
    'dash_dot': A line with alternating dashes and dots.
    'dash_dot_dot': A line with alternating dashes and two dots.
    'long_dash': A line with long dashes.
    'long_dash_dot': A line with long dashes and a dot.
    'long_dash_dot_dot': A line with long dashes and two dots.

    pattern options
    percent_xx
    dark_downward_diagonal
    dark_horizontal
    dark_upward_diagonal
    dark_vertical
    dashed_downward_diagonal
    dashed_horizontal
    dashed_upward_diagonal
    dashed_vertical
    diagonal_brick
    diagonal_cross
    pattern_divot
    dotted_diamond
    dotted_grid
    downward_diagonal
    horizontal
    horizontal_brick
    large_checker_board
    large_confetti
    large_grid
    light_downward_diagonal
    light_horizontal
    light_upward_diagonal
    light_vertical
    mixed_pattern
    narrow_horizontal
    narrow_vertical
    outlined_diamond
    plaid
    shingle
    small_checker_board
    small_confetti
    small_grid
    solid_diamond
    sphere
    trellis
    upward_diagonal
    vertical
    wave
    weave
    wide_downward_diagonal
    wide_upward_diagonal
    zig_zag

    marker options
    'none': No marker.
    'circle': Circular markers.
    'square': Square markers.
    'diamond': Diamond-shaped markers.
    'triangle': Triangle markers.
    'x': X-shaped markers.
    'plus': Plus-shaped markers.
    'dash': Dash-shaped markers.
    'star': Star-shaped markers.
    """

    if 'graph_type_1' in graph_specs:
        graph_type_1 = graph_specs['graph_type_1']
    else:
        graph_type_1 = {'type': 'line'}

    if 'graph_type_2' in graph_specs:
        assert 'graph_type_2_cols' in graph_specs, 'No columns for graph_type_2 given'
        graph_type_2 = graph_specs['graph_type_2']
        graph_type_2_cols = graph_specs['graph_type_2_cols']
    else:
        graph_type_2 = {}
        graph_type_2_cols = []

    charts = {}

    # skip last or first n rows
    if 'skip_last_n_rows' in graph_specs:
        max_row = len(df) - graph_specs['skip_last_n_rows']
    else:
        max_row = len(df)
    if 'skip_first_n_rows' in graph_specs:
        min_row = graph_specs['skip_first_n_rows']
    else:
        min_row = 1

    # handle MultiIndex columns
    if isinstance(df.index, pd.MultiIndex):
        nr_of_ind_cols = len(df.index.levels)
    else:
        nr_of_ind_cols = 1

    # select columns to plot
    if 'cols_to_include' in graph_specs:
        cols = graph_specs['cols_to_include']
    else:
        cols = df.columns

    # add columns to chart
    for series_nr, col in enumerate(cols):
        try:
            col_nr = df.columns.get_loc(col)

            # set specifications for plotting column: [sheetname, first_row, first_col, last_row, last_col]
            series_specs = {
                'name': [dfname, 0, col_nr + 1],
                'categories': [dfname, min_row, 0, max_row, nr_of_ind_cols - 1],
                'values': [dfname, min_row, col_nr + 1, max_row, col_nr + 1],
            }
            if 'trendline' in graph_specs and col in graph_specs['trendline']:
                series_specs['trendline'] = {
                    'type': graph_specs['trendline'][col],
                    'line': {
                        'color': graph_specs['column_styles'][col]['marker']['line']['color'],
                        'width': 1,
                        'dash_type': 'long_dash',
                    },
                }
            # if custom styles for column given, apply
            if 'column_styles' in graph_specs and col in graph_specs['column_styles']:
                col_style = graph_specs['column_styles'][col]
                series_specs = {
                    **series_specs,
                    **col_style,
                }

            if col not in graph_type_2_cols:
                # Create chart for graph type
                if 'graph_type_1' not in charts.keys():
                    charts['graph_type_1'] = workbook.add_chart(graph_type_1)
                # Add a data series to the chart
                charts['graph_type_1'].add_series(series_specs)
            else:
                # Create chart for graph type
                if 'graph_type_2' not in charts.keys():
                    charts['graph_type_2'] = workbook.add_chart(graph_type_2)
                if 'y2_axis' in graph_specs:
                    series_specs['y2_axis'] = graph_specs['y2_axis']
                # Add a data series to the chart
                charts['graph_type_2'].add_series(series_specs)

        except KeyError:
            print(f"Column {col} not found in dataframe {dfname}")

    chart1 = charts['graph_type_1']

    # Combine charts
    if len(charts.keys()) > 1:
        chart2 = charts['graph_type_2']
        chart1.combine(chart2)

    # set font size and titles for chart
    font_size = 10
    if 'font_size' in graph_specs:
        font_size = graph_specs['font_size']
    if 'title' in graph_specs:
        chart1.set_title \
            ({'name': graph_specs['title'], 'name_font': {'size': font_size + 1, 'bold': False}})
    if 'x_axis_title' in graph_specs:
        # Add x-axis label
        chart1.set_x_axis \
            ({'name': graph_specs['x_axis_title'], 'name_font': {'size': font_size, 'bold': False}})
    if 'y_axis_title' in graph_specs:
        # Add y-axis label
        chart1.set_y_axis \
            ({'name': graph_specs['y_axis_title'], 'name_font': {'size': font_size, 'bold': False}})

    # customize legend
    legend_specs = {
        'font': {'size': font_size, 'bold': False},
    }

    # handle columns that should be plotted but not shown on legend
    # (e.g. helper variables for ts-plotting)
    if 'delete_series_from_legend' in graph_specs:
        legend_specs['delete_series'] = graph_specs['delete_series_from_legend']

    chart1.set_legend(legend_specs)

    # add chart to the worksheet with given offset values at the given positions or at default values
    if 'position' in graph_specs:
        position = graph_specs['position']
    else:
        position = f'B{3 + 15 * graph_nr}'
    if 'x_y_scale' in graph_specs:
        x_y_scale = graph_specs['x_y_scale']
    else:
        x_y_scale = [1.0, 1.0]

    worksheet.insert_chart(position, chart1, {'x_scale': x_y_scale[0], 'y_scale': x_y_scale[1]})


def save_xlsx_wb(out_path: str, df_dict: dict, graphs: dict = None):
    """
    Write one Excel workbook containing multiple sheets (Pandas DataFrames) and optional charts.

    This helper exports each DataFrame in ``df_dict`` to its own worksheet (sheet name = dict key),
    using the ``xlsxwriter`` engine. If ``graphs`` is provided, one or more XlsxWriter charts are
    inserted into selected worksheets.

    Notes on data export
    --------------------
    - The DataFrame index is written to Excel (Pandas default for ``to_excel``), but any *index name*
      is removed (``df.index.name = None``) to avoid an extra header label.
    - If a DataFrame has a MultiIndex for *columns*, the columns are flattened via
      ``df.columns.to_flat_index()`` before exporting (so chart column lookups work reliably).
    - If a DataFrame has a MultiIndex for the *index*, all index levels are written as separate
      index columns in Excel and are used as the chart category range.

    Parameters
    ----------
    out_path:
        Output ``.xlsx`` file path.
    df_dict:
        Mapping ``{sheet_name: dataframe}``. Each key is converted to ``str`` and used as the worksheet name.
    graphs:
        Optional chart specification mapping per worksheet:

        ``{sheet_name: [graph_specs, graph_specs, ...]}``

        Each ``graph_specs`` entry describes one chart "slot" to insert into the given worksheet.
        Within one slot, the function can create a primary chart (``graph_type_1``) and optionally
        a secondary chart (``graph_type_2``) and then *combine* them.

        Minimal example::

            graphs = {
                "my_sheet": [
                    {
                        "graph_type_1": {"type": "line"},
                        "cols_to_include": ["col_a", "col_b"],
                        "position": "B3",
                        "x_y_scale": [2.0, 1.0],
                    }
                ]
            }

        Supported ``graph_specs`` keys
        ------------------------------
        Chart types
            - ``graph_type_1``: dict passed to ``workbook.add_chart(...)`` (default: ``{"type": "line"}``)
            - ``graph_type_2``: optional dict passed to ``workbook.add_chart(...)``
            - ``graph_type_2_cols``: required if ``graph_type_2`` is set; columns that should be plotted on chart 2
            - ``y2_axis``: optional (typically ``1``) to plot selected series on the secondary axis

        Data selection
            - ``cols_to_include``: list of columns to plot. If omitted, all DataFrame columns are used.
            - ``skip_first_n_rows``: skip N first data rows for chart ranges (default: 1; row 0 is header)
            - ``skip_last_n_rows``: skip N last data rows for chart ranges (default: 0)

        Placement / sizing
            - ``position``: Excel cell anchor like ``"B3"`` (default: ``"B{3 + 15*graph_nr}"``)
            - ``x_y_scale``: ``[x_scale, y_scale]`` passed to ``worksheet.insert_chart`` (default: ``[1.0, 1.0]``)

        Titles and axes
            - ``title``: chart title string
            - ``x_axis_title`` / ``y_axis_title``: axis label strings
            - ``font_size``: base font size used for title/axis/legend (default: 10)

        Styling and legend
            - ``column_styles``: mapping ``{column_name: series_style_dict}`` merged into the series specs passed
              to ``chart.add_series(...)``. Typical keys: ``line``, ``fill``, ``pattern``, ``marker``.
            - ``delete_series_from_legend``: list of series indices to hide from the legend (passed through to
              ``chart.set_legend({'delete_series': ...})``)

        Trendlines
            - ``trendline``: mapping ``{column_name: trendline_type}`` (e.g. ``"linear"``).
              The trendline line color is taken from
              ``graph_specs['column_styles'][col]['marker']['line']['color']`` (so if you use trendlines,
              ensure that path exists in your styles).

    Behavior and limitations
    ------------------------
    - For each plotted column, the function looks up the column index via ``df.columns.get_loc(col)``.
      If a specified column is missing, it prints a message and continues.
    - Chart categories always refer to the exported index columns (first worksheet columns).
      For MultiIndex indices, the category range spans all index-level columns.
    - If no valid series are added for ``graph_type_1``, chart creation will fail (because the code
      expects a primary chart to exist).

    Returns
    -------
    None
        Writes the workbook to disk and closes the writer.
    """

    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")

    for dfname, df in df_dict.items():
        dfname = str(dfname)
        df.index.name = None # drop index name
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.to_flat_index()  # flatten multiindex for columns
        df.to_excel(writer, sheet_name=dfname)

        if graphs and dfname in graphs.keys():

            # access the XlsxWriter workbook and worksheet objects from the dataframe.
            workbook = writer.book
            worksheet = writer.sheets[dfname]

            # add charts to worksheet
            for graph_nr, graph_specs in enumerate(graphs[dfname]):
                add_chart_to_worksheet(workbook, worksheet, graph_nr, graph_specs, df, dfname)

    writer.close()

    print(f"wrote {out_path}")
