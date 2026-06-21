from django.urls import path
from . import views

from django.urls import path
from . import views

urlpatterns = [
    # Home / Landing Page
    path('', views.home_redirect_view, name='home'),

    # Authentication
    # These names must match the functions in views.py exactly:
    # def register(request): ...
    path('register/', views.register, name='register'), 
    
    # def otp_verify(request): ...
    path('otp-verify/', views.otp_verify, name='otp_verify'),
    
    # login_view is correct because we defined it as: login_view = CustomLoginView.as_view()
    path('login/', views.login_view, name='login'),
    
    # def custom_logout_view(request): ...
    path('logout/', views.custom_logout_view, name='custom_logout'),

    # Dashboard & Analyzer
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('analyzer/', views.analyzer_view, name='analyzer'),
    
    # API Endpoint
    # def predict_api(request): ...
    path('predict/', views.predict_api, name='predict_api'),

    # Metrics Pages
    path('metrics/confusion-matrix/', views.metrics_cm_view, name='metrics_cm'),
    path('metrics/accuracy/', views.metrics_accuracy_view, name='metrics_accuracy'),
    path('metrics/loss/', views.metrics_loss_view, name='metrics_loss'),

    # Static Pages
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
]