"""
@DATE: 2022/8/22
@Author  : ld
"""
from flask import Flask, render_template

from conf import HOST, PORT, DEBUG

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)

