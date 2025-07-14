import panel as pn
from youtube_dashboard import get_interactive_dashboard_app

pn.extension('plotly')
print("🔧 MAIN.PY SERVER STARTING UP...")

def app():
    return get_interactive_dashboard_app()

if __name__ == "__main__":
    pn.serve(
        {'/main': app},
        address="0.0.0.0",
        port=10000,
        allow_websocket_origin=["siads521-youtube-dashboard.onrender.com"]
    )
