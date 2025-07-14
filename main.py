import panel as pn
from youtube_dashboard import create_interactive_dashboard

pn.extension('plotly')
print("MAIN.PY SERVER STARTING UP...")

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

if __name__ == "__main__":
    pn.serve(
        {'/main': app},
        port=10000,
        address="0.0.0.0",
        allow_websocket_origin=["siads521-youtube-dashboard.onrender.com"]
    )
