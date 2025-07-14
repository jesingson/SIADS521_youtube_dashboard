import panel as pn
pn.extension('plotly')

from youtube_dashboard import create_interactive_dashboard

def app():
    return create_interactive_dashboard(
        metric='log_views',
        catFilters=['Music'],
        countryFilters=['US', 'CA', 'GB'],
        bins=50,
        xAxisMetric='log_likes',
        xAxisDate='publish_time',
        periodRollup='Weekly'
    )

# This allows both local use (panel serve main.py) and Render use
dashboard = app()
dashboard.servable()

# Optional: keep this for direct script execution, but not required by Render
if __name__ == "__main__":
    pn.serve(
        {'/': app},
        port=10000,
        address="0.0.0.0",
        allow_websocket_origin=["siads521-youtube-dashboard.onrender.com"]
    )