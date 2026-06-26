try:
    from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
except ImportError as exc:
    raise ImportError(
        "Flask-Login is not installed in this Python interpreter. "
        "Activate the project's .venv and install dependencies, or run: .\\.venv\\Scripts\\python.exe -m pip install Flask-Login"
    ) from exc

try:
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError as exc:
    raise ImportError(
        "Werkzeug is not installed in this Python interpreter. "
        "Activate the project's .venv and install dependencies, or run: .\\.venv\\Scripts\\python.exe -m pip install werkzeug"
    ) from exc

login_manager = LoginManager()
login_manager.login_view = "profile"


def init_auth(app):
    login_manager.init_app(app)


__all__ = [
    "login_manager",
    "init_auth",
    "UserMixin",
    "login_user",
    "logout_user",
    "current_user",
    "login_required",
    "generate_password_hash",
    "check_password_hash"
]
