import panel as pn
from youtube_dashboard import create_interactive_dashboard

pn.extension('plotly')

def app():
    dashboard = create_interactive_dashboard(
        metric='log_views',
        catFilters=['Music'],
        countryFilters=['US', 'CA', 'GB'],
        bins=50,
        xAxisMetric='log_likes',
        xAxisDate='publish_time',
        periodRollup='Weekly'
    )
    return dashboard

# This tells Render how to serve the app
if __name__ == "__main__":
    pn.serve(
        {'/': app},  # Serve the dashboard function
        port=10000,
        address="0.0.0.0",
        allow_websocket_origin=["siads521-youtube-dashboard.onrender.com"]
    )
