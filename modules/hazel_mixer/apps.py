from django.apps import AppConfig


class HazelMixerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    label = "hazel_mixer"
    name = "modules." + label