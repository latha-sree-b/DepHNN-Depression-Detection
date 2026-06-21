# --- All Imports ---
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.contrib.auth import login, logout, get_user_model
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .forms import CustomUserCreationForm

# --- THIS WAS MISSING ---
from .models import PredictionLog 
# ------------------------

import random
import numpy as np
import tensorflow as tf
import mne
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time

# --- Preprocessing Constants ---
SAMPLING_FREQ = 256
FINAL_CHANNEL_NAMES_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]
CHANNEL_RENAME_MAP = {
    'EEG Fp1-LE': 'Fp1', 'EEG Fp2-LE': 'Fp2', 'EEG F3-LE': 'F3', 'EEG F4-LE': 'F4',
    'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3', 'EEG P4-LE': 'P4',
    'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7', 'EEG F8-LE': 'F8',
    'EEG T3-LE': 'T7', 'EEG T4-LE': 'T8', 'EEG T5-LE': 'P7', 'EEG T6-LE': 'P8',
    'EEG Fz-LE': 'Fz', 'EEG Cz-LE': 'Cz', 'EEG Pz-LE': 'Pz'
}
FILTER_L_FREQ = 0.5
FILTER_H_FREQ = 45
WINDOW_DURATION = 1.0

# --- Load Model ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'DepHNN_model_final.keras')

print(" * Loading Keras model for Django...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" * Model loaded successfully!")
except Exception:
    print(" * Model file not found. Prediction will fail.")
    model = None


# --- Preprocessing Function ---
def preprocess_single_file(filepath):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

        raw.rename_channels(CHANNEL_RENAME_MAP)
        available_channels = [ch for ch in FINAL_CHANNEL_NAMES_19 if ch in raw.info['ch_names']]
        raw.pick(available_channels)

        if raw.info['sfreq'] != SAMPLING_FREQ:
            raw.resample(SAMPLING_FREQ, verbose=False)

        raw.filter(FILTER_L_FREQ, FILTER_H_FREQ, fir_design='firwin', verbose=False)

        n_components = len(raw.info['ch_names'])
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
        ica.fit(raw)
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        ica.exclude.extend(eog_inds)
        muscle_inds, _ = ica.find_bads_muscle(raw)
        if muscle_inds:
            ica.exclude.extend(muscle_inds)
        ica.apply(raw, verbose=False)

        scaler = StandardScaler()
        data_norm = scaler.fit_transform(raw.get_data().T).T
        raw_norm = mne.io.RawArray(data_norm, raw.info, verbose=False)

        epochs = mne.make_fixed_length_epochs(raw_norm, duration=WINDOW_DURATION, overlap=0.5, preload=True, verbose=False)
        return epochs.get_data().astype(np.float32)

    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None

User = get_user_model()

# =========================================
#  CORE VIEWS
# =========================================

def home_redirect_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'home.html')

@login_required
def dashboard_view(request):
    return render(request, 'dashboard.html')

from django.contrib.auth.views import LoginView
class CustomLoginView(LoginView):
    template_name = 'registration/login.html'
login_view = CustomLoginView.as_view()


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            
            otp_code = random.randint(100000, 999999)
            request.session['otp_code'] = otp_code
            request.session['user_id'] = user.id
            request.session['otp_creation_time'] = time.time()
            
            subject = 'Your DepHNN Account Verification Code'
            message = f'Your 6-digit verification code is: {otp_code}. Expires in 5 mins.'
            
            send_mail(subject, message, 'noreply@dephnn.com', [user.email], fail_silently=False)
            messages.success(request, 'Registration successful! Please check your email for the OTP.')
            return redirect('otp_verify')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def otp_verify(request):
    if 'user_id' not in request.session:
        messages.error(request, 'Session expired. Please register again.')
        return redirect('register')

    if request.method == 'POST':
        user_otp = request.POST.get('otp')
        session_otp = request.session.get('otp_code')
        creation_time = request.session.get('otp_creation_time')
        
        # Expiry Check
        if (time.time() - creation_time) > 300:
            del request.session['otp_code']
            del request.session['user_id']
            del request.session['otp_creation_time']
            try:
                User.objects.get(id=request.session.get('user_id')).delete()
            except: pass
            messages.error(request, 'OTP has expired. Please register again.')
            return redirect('register')
        
        if not user_otp:
            messages.error(request, 'Please enter the code.')
        elif session_otp is not None and user_otp.isdigit() and int(user_otp) == session_otp:
            user_id = request.session.get('user_id')
            try:
                user = User.objects.get(id=user_id)
                user.is_active = True
                user.save()
                login(request, user)
                del request.session['otp_code']
                del request.session['user_id']
                del request.session['otp_creation_time']
                messages.success(request, 'Verification successful!')
                return redirect('dashboard')
            except User.DoesNotExist:
                 messages.error(request, 'User error.')
                 return redirect('register')
        else:
            messages.error(request, 'Invalid OTP.')
    
    return render(request, 'registration/otp_verify.html')

@login_required
def custom_logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('home')

# =========================================
#  STATIC & METRIC VIEWS
# =========================================

@login_required
def analyzer_view(request): return render(request, 'analyzer.html')

@login_required
def metrics_cm_view(request): return render(request, 'metrics/confusion_matrix.html')

@login_required
def metrics_accuracy_view(request): return render(request, 'metrics/accuracy.html')

@login_required
def metrics_loss_view(request): return render(request, 'metrics/loss.html')

def about_view(request): return render(request, 'about.html')
def contact_view(request): return render(request, 'contact.html')

# =========================================
#  API ENDPOINT
# =========================================

@login_required 
def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method"}, status=405)

    file = request.FILES.get('file')
    if not file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    temp_path = "temp_upload_file.edf"
    with open(temp_path, 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    
    try:
        print(f"Processing file: {file.name}")
        windows = preprocess_single_file(temp_path)
        
        if windows is None:
            return JsonResponse({"error": "Preprocessing failed."}, status=500)

        predictions = model.predict(windows, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        depressed_votes = np.sum(predicted_classes == 1)
        total_votes = len(predicted_classes)
        confidence = depressed_votes / total_votes

        # --- Advice Logic ---
        if confidence > 0.5:
            prediction = "Depressed"
            final_conf = confidence
            if confidence >= 0.80:
                 disorders = ["Major Depressive Disorder"]
            else:
                 disorders = ["Depressed (Mild/Moderate symptoms detected)"]
            
            advice_list = [
                "⚠️ **Consult a Professional:** We strongly recommend sharing these results with a psychologist or psychiatrist for a formal diagnosis.",
                "🗣️ **Stay Connected:** Do not isolate yourself. Reach out to trusted friends or family members.",
                "😴 **Sleep Routine:** Try to maintain a consistent sleep schedule, as sleep impacts mood significantly.",
                "🏃 **Physical Activity:** Light exercise (like walking) can help release endorphins and improve mood."
            ]
        else:
            prediction = "Normal"
            final_conf = 1.0 - confidence
            disorders = ["Healthy"]
            advice_list = [
                "✅ **All Good:** Your EEG patterns appear within the healthy range.",
                "🌟 **Maintain Wellness:** Continue your current healthy lifestyle habits.",
                "🧘 **Mental Hygiene:** Regular stress management and mindfulness are great for long-term brain health."
            ]

        # --- Save to Database ---
        PredictionLog.objects.create(
            user=request.user,
            file_name=file.name,
            prediction_result=prediction,
            confidence_score=round(final_conf, 3)
        )

        response_data = {
            "Prediction": prediction,
            "Confidence": round(final_conf, 3),
            "Detected_Disorders": disorders,
            "Advice": advice_list, # Send advice to frontend
            "Details": {
                "total_windows": int(total_votes),
                "depressed_windows": int(depressed_votes),
                "healthy_windows": int(total_votes - depressed_votes)
            }
        }
        return JsonResponse(response_data)

    except Exception as e:
         print(f"Prediction Error: {e}")
         return JsonResponse({"error": str(e)}, status=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)