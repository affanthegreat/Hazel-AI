from django.apps import AppConfig


class EdenExperienceEngineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    label = "eden_experience_engine"
    name = "modules." + label
