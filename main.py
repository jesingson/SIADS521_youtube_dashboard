import panel as pn
from youtube_dashboard import create_interactive_dashboard  # Or whatever function you use

# Call this once to initialize widgets and dashboard
pn.extension('plotly')

# Initial render (you can customize inputs here or use default ones)
dashboard = create_interactive_dashboard(
    metric='log_views',
    catFilters=['Music'],
    countryFilters=['US', 'CA', 'GB'],
    bins=50,
    xAxisMetric='log_likes',
    xAxisDate='publish_time',
    periodRollup='Weekly'
)

dashboard.servable()
