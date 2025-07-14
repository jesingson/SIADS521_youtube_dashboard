import panel as pn
import numpy as np
import plotly.graph_objs as go

pn.extension('plotly')

# Widget: How many points in the sine wave
points_slider = pn.widgets.IntSlider(name="Number of Points", start=10, end=1000, step=10, value=100)

# Button
submit_button = pn.widgets.Button(name="Submit", button_type="primary")

# Output container
plot_container = pn.Column()

# Callback
def on_submit(event):
    print("✅ Submit fired with value:", points_slider.value)
    x = np.linspace(0, 2*np.pi, points_slider.value)
    y = np.sin(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
    fig.update_layout(title="Sine Wave", height=400)

    plot_container.objects = [pn.pane.Plotly(fig)]

submit_button.on_click(on_submit)

# Trigger initial render
on_submit(None)

# Final layout
app = pn.Column(
    pn.pane.Markdown("### Minimal Submit Button Dashboard"),
    points_slider,
    submit_button,
    pn.Spacer(height=10),
    plot_container,
    sizing_mode="stretch_width"
)

# Entry point
def main_app():
    return app

if __name__ == "__main__":
    pn.serve(
        {'/main': main_app},
        port=10000,
        address="0.0.0.0",
        allow_websocket_origin=["siads521-youtube-dashboard.onrender.com"]
    )
