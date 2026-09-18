import pytest


streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_reset_conversation_clears_state_before_preset_widget_reruns():
    app_test = streamlit_testing.AppTest.from_file("app.py").run()
    assert not app_test.exception

    app_test.button[0].click().run()

    assert not app_test.exception
    assert app_test.selectbox[0].value == "Custom"
