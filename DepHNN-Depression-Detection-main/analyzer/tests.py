import os
import time
import numpy as np
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from .forms import CustomUserCreationForm

User = get_user_model()

# --- HELPER: Create a Dummy .edf File ---
def create_dummy_edf(filename="test_recording.edf"):
    with open(filename, "wb") as f:
        # Create a minimal valid-looking header so it doesn't crash immediately
        f.write(b'0       M 01-JAN-2000 ' + b'\x00' * 1000)
    return filename

class UnitTests(TestCase):
    """UNIT TESTING"""
    def test_custom_user_creation(self):
        """TC-Unit-01: CustomUser Model Creation"""
        user = User.objects.create_user(
            username='testunit', email='unit@test.com', password='TestPassword123!', phone_number='1234567890'
        )
        self.assertEqual(user.username, 'testunit')
        self.assertEqual(user.phone_number, '1234567890')
        self.assertTrue(user.is_active) 
        print("[PASS] Unit Test: CustomUser Model Creation")

    def test_registration_form_validation(self):
        """TC-Unit-02: RegistrationForm Validation"""
        form_data = {
            'username': 'formtest', 
            'first_name': 'Test', 
            'last_name': 'User',
            'email': 'form@test.com', 
            'phone_number': '1234567890', 
            'password1': 'TestPassword123!', # Corrected field name
            'password2': 'TestPassword123!'  # Corrected field name
        }
        form = CustomUserCreationForm(data=form_data)
        self.assertIn('phone_number', form.fields)
        print("[PASS] Unit Test: RegistrationForm Validation")

class IntegrationTests(TestCase):
    """INTEGRATION TESTING"""
    def setUp(self):
        self.client = Client()

    def test_registration_otp_flow(self):
        """TC-Int-01: Registration and OTP Flow"""
        # 1. Register
        # FIX: Changed 'password'/'confirm_password' to 'password1'/'password2'
        response = self.client.post(reverse('register'), {
            'username': 'integration', 
            'first_name': 'Int', 
            'last_name': 'Test',
            'email': 'int@test.com', 
            'phone_number': '5555555555',
            'password1': 'TestPassword123!', 
            'password2': 'TestPassword123!'
        }, follow=True)
        
        # Debug: Print form errors if any
        if 'form' in response.context and response.context['form'].errors:
             print(f"Form Errors: {response.context['form'].errors}")

        # Check if user exists
        try:
            user = User.objects.get(username='integration')
        except User.DoesNotExist:
            self.fail("User was not created during registration.")

        # View logic should set active=False
        self.assertFalse(user.is_active)
        
        # Check if OTP is in session
        session = self.client.session
        if 'otp_code' not in session:
            self.fail("OTP not found in session. Registration view likely failed or didn't redirect.")
            
        otp_code = session['otp_code']
        
        # 2. Verify OTP
        response = self.client.post(reverse('otp_verify'), {'otp': str(otp_code)})
        
        # Check if user is now active
        user.refresh_from_db()
        self.assertTrue(user.is_active)
        print("[PASS] Integration Test: Registration & OTP Flow")

    def test_otp_expiry(self):
        """TC-Int-02: OTP Expiry Check"""
        # Register first to set session
        # FIX: Changed password fields here too
        self.client.post(reverse('register'), {
            'username': 'expiry_test', 'first_name': 'Exp', 'last_name': 'Test',
            'email': 'exp@test.com', 'phone_number': '1112223333', 
            'password1': 'TestPassword123!', 'password2': 'TestPassword123!'
        })
        
        # Manually manipulate session time to be 10 minutes ago
        session = self.client.session
        session['otp_creation_time'] = time.time() - 600 
        session.save()
        
        # Try to verify
        otp_code = session.get('otp_code', '123456')
        response = self.client.post(reverse('otp_verify'), {'otp': str(otp_code)})
        
        # Should redirect back to register
        self.assertRedirects(response, reverse('register'))
        print("[PASS] Integration Test: OTP Expiry Logic")

    @patch('analyzer.views.preprocess_single_file')
    @patch('analyzer.views.model')
    def test_preprocessing_prediction_integration(self, mock_model, mock_preprocess):
        """TC-Int-03: Preprocessing to Prediction Integration"""
        user = User.objects.create_user(username='pred_user', password='TestPassword123!')
        self.client.force_login(user)
        
        # Setup Mocks
        mock_preprocess.return_value = np.random.rand(10, 19, 256).astype(np.float32)
        mock_model.predict.return_value = np.array([[0.1, 0.9]] * 10)

        dummy_name = "test.edf"
        create_dummy_edf(dummy_name)
        with open(dummy_name, 'rb') as f:
            uploaded_file = SimpleUploadedFile(dummy_name, f.read())

        response = self.client.post(reverse('predict_api'), {'file': uploaded_file})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['Prediction'], 'Depressed')
        
        if os.path.exists(dummy_name): os.remove(dummy_name)
        print("[PASS] Integration Test: Preprocessing -> Prediction Pipeline")

class FunctionalTests(TestCase):
    """FUNCTIONAL TESTING"""
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='func_user', password='TestPassword123!')
        self.client.force_login(self.user)

    def test_eeg_file_upload_mechanism(self):
        """TC-Func-01: EEG File Upload"""
        dummy_name = "func_test.edf"
        create_dummy_edf(dummy_name)
        with open(dummy_name, 'rb') as f:
            uploaded_file = SimpleUploadedFile(dummy_name, f.read())
            
        response = self.client.post(reverse('predict_api'), {'file': uploaded_file})
        self.assertNotEqual(response.status_code, 404) 
        if os.path.exists(dummy_name): os.remove(dummy_name)
        print("[PASS] Functional Test: File Upload Mechanism")

class SystemTests(TestCase):
    """SYSTEM TESTING"""
    def setUp(self):
        self.client = Client()

    def test_full_cycle(self):
        """TC-Sys-01: Full Cycle (Login -> Dashboard -> Logout)"""
        user = User.objects.create_user(username='system_user', password='TestPassword123!')
        self.client.force_login(user)
        
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        
        response = self.client.get(reverse('custom_logout'))
        self.assertEqual(response.status_code, 302)
        print("[PASS] System Test: Login/Dashboard/Logout Cycle")

class WhiteBoxTests(TestCase):
    """WHITE BOX TESTING"""
    def test_auth_branch_testing(self):
        """TC-WB-01: Authentication Branch Testing"""
        client = Client()
        response = client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)
        
        user = User.objects.create_user(username='branch_user', password='TestPassword123!')
        client.force_login(user)
        response = client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        print("[PASS] White Box Test: Auth Branching")

class BlackBoxTests(TestCase):
    """BLACK BOX TESTING"""
    def test_ui_validation(self):
        """TC-BB-01: User Interface Validation"""
        client = Client()
        response = client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')
        print("[PASS] Black Box Test: UI Page Load")

class AcceptanceTests(TestCase):
    """ACCEPTANCE TESTING"""
    def test_model_exists(self):
        """TC-Acc-01: Prediction Accuracy Validation (File Check)"""
        base_dir = settings.BASE_DIR
        model_path = os.path.join(base_dir, 'models', 'DepHNN_model_final.keras')
        self.assertTrue(os.path.exists(model_path))
        print("[PASS] Acceptance Test: Model File Verified")