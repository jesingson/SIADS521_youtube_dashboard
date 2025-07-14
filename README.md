# YouTube Trending Dashboard

This project is an interactive data dashboard built using Panel, Plotly, and Jupyter, 
visualizing trends from a subset of YouTube trending videos in the US, CA, and GB — 
based on the publicly available [YouTube Trending Dataset on Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new).

## 📁 Project Repository
You can find all source code, notebooks, and deployment files in the public GitHub repository:
🔗 github.com/jesingson/SIADS521_youtube_dashboard

This includes:
* The original Jupyter Notebook (.ipynb) with full analysis and documentation
* A cleaned Python script version (youtube_dashboard.py) ready for cloud deployment
* The deployment entry point (main.py)
* Data truncation and environment setup details

## 💡 Features
- Interactive histograms, violinplots, scatterplots, and time series charts
- Controls for filtering by metric, category, country, and time aggregation
- Exportable as a standalone HTML file or runnable on a Render deployment

# 🧪 Try It Live
Interact with the live dashboard here:
👉 https://siads521-youtube-dashboard.onrender.com/main

A walkthrough video will be available soon: [https://umich.zoom.us/rec/share/kdu90G-C-2ekp-wK70mrD287c-R4rbShmSGyxtNEznLqWwGB0Z2d2L05ipFaTYYI.1uAL1u0sHCSXzpvz]

## 📁 File Guide

| File/Folder               | Purpose                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `youtube_dashboard.ipynb` | **Main notebook** – includes full write-up, documentation, code, and an interactive version of the dashboard (for local exploration) |
| `youtube_dashboard.py`    | Converted `.py` script used for deployment                                                                                           |
| `main.py`                 | Entrypoint script for serving the dashboard on Render                                                                                |
| `youtube_data/`           | Folder containing preprocessed YouTube trending datasets                                                                             |
| `requirements.txt`        | Python dependencies                                                                                                                  |
| `README.md`               | This file                                                                                                                            |


## ⚙️ Setup Locally
pip install -r requirements.txt
panel serve main.py --address 0.0.0.0 --port 10000

You must also include the correct --allow-websocket-origin flag if deploying remotely.

### 🚀 Render Deployment Notes
To get the .py version working on Render, the following changes were made:

* ❌ Removed all Jupyter-specific shell commands (!pip, etc.)
* 🔁 Replaced all display(...) calls with print(...)
* 📉 Limited the size of the loaded dataset using pd.read_csv(..., nrows=20000) to stay within Render's 512MB memory limit
* 🔧 Refactored widget creation into a new function: get_interactive_dashboard_app()
* 🧩 Created main.py to act as the proper entrypoint for panel serve
* 🧵 Reorganized dashboard layout to ensure the Submit button re-renders charts correctly

## 📬 Notes for Reviewers
* If you're reviewing this project, start by exploring the .ipynb file to see the full methodology, EDA, and build process.
* You can skip the .py files unless you're specifically interested in the deployable architecture.
* The easiest way to experience the dashboard is through the public Render link above.

## ⚠️ Notes on Hosting and Availability
This dashboard is hosted on Render, a cloud platform that provides free-tier web service hosting. When visiting the public dashboard link:

🔗 https://siads521-youtube-dashboard.onrender.com/main

You may occasionally see a 502 Bad Gateway error. This can happen because:
* Render spins down inactive services after ~15 minutes to conserve resources.
* When the service is restarted ("cold start"), it can take up to a minute for the dashboard to become available again.
* Memory usage spikes during data loading or chart generation can also cause temporary restarts on the free plan.

🕒 What to do:
* Just wait 30–60 seconds and refresh the page. The service should recover automatically.
* If the issue persists, try again in a few minutes or contact me via the information below.