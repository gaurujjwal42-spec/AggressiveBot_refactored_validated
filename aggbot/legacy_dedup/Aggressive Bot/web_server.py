from flask import Flask, render_template, request
import math
import database

app = Flask(__name__)

# Ensure the templates folder is created for render_template to work
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

@app.route('/')
def history():
    """Displays the trade history with pagination."""
    page = request.args.get('page', 1, type=int)
    limit = 15 # Trades per page

    # Fetch active positions and trade history
    active_positions = database.load_positions()
    trades, total_trades = database.get_trade_history_paginated(page, limit)
    total_pages = math.ceil(total_trades / limit)

    return render_template(
        'history.html',
        trades=trades,
        active_positions=active_positions.values(),
        current_page=page,
        total_pages=total_pages
    )

if __name__ == '__main__':
    app.run(debug=False, port=5001)