from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, PredictionLog

# 1. Custom User Admin
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['username', 'email', 'phone_number', 'is_active', 'is_staff', 'date_joined']
    fieldsets = UserAdmin.fieldsets + (
        ('Extra Info', {'fields': ('phone_number',)}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (None, {'fields': ('phone_number',)}),
    )

# 2. Prediction Log Admin (NEW)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'prediction_result', 'confidence_score', 'file_name', 'timestamp']
    list_filter = ['prediction_result', 'timestamp']
    search_fields = ['user__username', 'file_name']

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(PredictionLog, PredictionLogAdmin)