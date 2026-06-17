from flask_sqlalchemy import SQLAlchemy

# Shared SQLAlchemy object for the application.
db = SQLAlchemy()


def init_db(app):
    db.init_app(app)


__all__ = ["db", "init_db"]
