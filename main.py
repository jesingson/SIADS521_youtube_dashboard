import panel as pn
pn.extension('plotly')

# Import the notebook as a module (requires jupyter, nbconvert, etc.)
pn.serve("youtube_dashboard.ipynb", 
         port=8000, 
         address="0.0.0.0", 
         allow_websocket_origin=["*"],
         autoreload=True,
         show=False)
