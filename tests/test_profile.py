import unittest

from app import app, db, User, Reminder


class ProfileModuleTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True, SQLALCHEMY_DATABASE_URI="sqlite:///:memory:")
        self.client = app.test_client()
        with app.app_context():
            db.drop_all()
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_register_sets_persistent_session_cookie(self):
        response = self.client.post(
            "/profile",
            data={
                "auth_action": "register",
                "email": "persist@example.com",
                "password": "StrongPass123",
                "confirm_password": "StrongPass123",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        set_cookie = response.headers.get("Set-Cookie", "")
        self.assertIn("remember_token=", set_cookie)
        self.assertIn("Expires=", set_cookie)

    def test_authenticated_user_can_create_reminder(self):
        self.client.post(
            "/profile",
            data={
                "auth_action": "register",
                "email": "reminder@example.com",
                "password": "StrongPass123",
                "confirm_password": "StrongPass123",
            },
            follow_redirects=False,
        )

        reminder_data = {
            "auth_action": "reminder",
            "title": "Daily Walk",
            "reminder_type": "exercise",
            "message": "Take a 20-minute walk today.",
            "reminder_datetime": "2026-07-02T08:00",
            "is_active": "on",
        }
        response = self.client.post("/profile", data=reminder_data, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Daily Walk", response.get_data(as_text=True))

        with app.app_context():
            reminders = Reminder.query.all()
            self.assertEqual(len(reminders), 1)
            self.assertEqual(reminders[0].title, "Daily Walk")


if __name__ == "__main__":
    unittest.main()
