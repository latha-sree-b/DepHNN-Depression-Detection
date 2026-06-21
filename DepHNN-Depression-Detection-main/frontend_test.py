import os
import time
import django
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- 1. SETUP DJANGO ENVIRONMENT ---
print("⚙️ Setting up Django environment...")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dephnn_django.settings')
django.setup()

from django.contrib.auth import get_user_model
User = get_user_model()

# --- 2. CREATE TEST USER ---
USERNAME = "selenium_auto_user"
PASSWORD = "TestPassword123!"

try:
    if not User.objects.filter(username=USERNAME).exists():
        print(f"Creating new test user: {USERNAME}")
        user = User.objects.create_user(
            username=USERNAME,
            email="auto@test.com",
            password=PASSWORD,
            phone_number="1234567890"
        )
        user.is_active = True 
        user.save()
    else:
        print(f"User {USERNAME} already exists. Resetting status.")
        user = User.objects.get(username=USERNAME)
        user.set_password(PASSWORD)
        user.is_active = True
        user.save()
    print("✅ Test user ready.")
except Exception as e:
    print(f"❌ Database Error: {e}")
    exit()

# --- 3. START SELENIUM ---
print("🚀 Starting Browser...")
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

try:
    # --- LOGIN FLOW ---
    print("\n--- Testing Login ---")
    driver.get("http://127.0.0.1:8000/login/")
    
    driver.find_element(By.NAME, "username").send_keys(USERNAME)
    driver.find_element(By.NAME, "password").send_keys(PASSWORD)
    
    submit_btn = driver.find_element(By.XPATH, "//button[@type='submit']")
    submit_btn.click()
    print("Submitted Login Form.")
    
    time.sleep(2)
    
    # --- FIX: Update the text we look for ---
    # Your new dashboard says "System Dashboard", not "Project Dashboard"
    if "System Dashboard" in driver.page_source:
        print("✅ SUCCESS: Logged in and reached Dashboard.")
    elif "Enter Your 6-Digit OTP" in driver.page_source:
        print("⚠️ NOTE: Redirected to OTP. (User was not set to active correctly).")
    else:
        print("❌ FAILURE: Login failed. Check credentials or page source.")
        print("Page text:", driver.find_element(By.TAG_NAME, "body").text[:100])
        raise Exception("Login failed")

    # --- UPLOAD FLOW ---
    print("\n--- Testing Navigation ---")
    driver.find_element(By.CSS_SELECTOR, "a.bento-card.main-tool").click() # Updated class selector
    time.sleep(1)
    
    if "Upload an .edf file" in driver.page_source:
        print("✅ SUCCESS: Navigated to Analyzer Tool.")
    else:
        print("❌ FAILURE: Could not find Analyzer page.")

   # --- LOGOUT FLOW ---
    print("\n--- Testing Logout ---")
    
    # Try to find the logout button (it might be 'logout-btn' or 'logout-btn-small')
    try:
        driver.find_element(By.CLASS_NAME, "logout-btn").click()
        print("Clicked 'logout-btn'.")
    except:
        driver.find_element(By.CLASS_NAME, "logout-btn-small").click()
        print("Clicked 'logout-btn-small'.")
    
    time.sleep(1)
    
    # Click Confirm Logout (Red button)
    logout_confirm = driver.find_element(By.CLASS_NAME, "modal-btn-danger")
    logout_confirm.click()
    
    time.sleep(1)
    
    if "DepHNN Analyzer" in driver.page_source and "Sign Up" in driver.page_source:
        print("✅ SUCCESS: Logged out and returned to Landing Page.")
    else:
        print("❌ FAILURE: Logout did not return to landing page.")
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")

finally:
    print("\nClosing browser...")
    driver.quit()