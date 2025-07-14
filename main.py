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

# IMPORTANT: This tells Render how to serve the app
pn.serve(app, title="YouTube Dashboard", port=8000, address="0.0.0.0")
