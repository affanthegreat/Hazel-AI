from django.apps import AppConfig


class HazelcommunicatorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    label = "hazelcommunicator"
    name = "modules." + label