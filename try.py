import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.io import curdoc

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))

# Create a plot
p = figure(title="Interactive Y-Range Plot", x_axis_label='X', y_axis_label='Y')
p.line('x', 'y', source=source, line_width=2)

# Create a slider to change the y-axis range
slider = Slider(start=0, end=10, value=1, step=0.1, title="Y-Range")

# Update function to change y-axis range
def update(attr, old, new):
    p.y_range.start = -slider.value
    p.y_range.end = slider.value

slider.on_change('value', update)

# Layout
layout = column(p, slider)

# Add the layout to the current document
curdoc().add_root(layout)

# Save and show the plot
output_file("interactive_y_range_plot.html")
show(layout)

