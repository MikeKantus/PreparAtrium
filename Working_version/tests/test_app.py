from preparatrium import app


def test_app_welcome_message():
    message = app.build_welcome_message("Ava")
    assert message == "Welcome to PreparAtrium, Ava!"
