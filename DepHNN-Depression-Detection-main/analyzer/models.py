from django.db import models
from django.contrib.auth.models import AbstractUser

# 1. Custom User Model (Existing)
class CustomUser(AbstractUser):
    phone_number = models.CharField(max_length=15, blank=True, null=True)

    def __str__(self):
        return self.username

# 2. Prediction Log Model (NEW - Matches your System Architecture)
class PredictionLog(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    prediction_result = models.CharField(max_length=50) # e.g., "Depressed"
    confidence_score = models.FloatField() # e.g., 0.98
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.prediction_result} ({self.timestamp})"