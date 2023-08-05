from django.apps import AppConfig


class EdenConxEngineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    label = "eden_conX_engine"
    name = "modules." + label
