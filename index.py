from core_app.app import app


if __name__ == "__main__":
    app.run(
        #host="0.0.0.0",   # listen on all interfaces
        port=8000,        # or any available port
        debug=True,       # turn on hot-reload in dev
        #dev_tools_hot_reload=True,  # more granular control
    )