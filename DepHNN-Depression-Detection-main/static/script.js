document.addEventListener("DOMContentLoaded", () => {

    // DOM elements
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileName');
    const modal = document.getElementById('logout-modal');
    const backdrop = document.getElementById('logout-modal-backdrop');
    let probChart = null;

    // Event Listener: Display filename when selected
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileNameDisplay.textContent = "Selected: " + e.target.files[0].name;
            }
        });
    }

    // Main Analysis Function
    window.analyze = async function(csrfToken) {
        if (!fileInput || fileInput.files.length === 0) {
            alert("Please select an .edf EEG file first!");
            return;
        }

        // 1. Update UI for loading state
        document.getElementById('loading').style.display = 'block';
        document.getElementById('results').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('analyzeBtn').textContent = 'Analyzing...';

        // 2. Prepare data for backend
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            // 3. Send file to Django backend
            const response = await fetch('/predict/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken
                },
                body: formData
            });
            
            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Analysis failed');

            // 4. Display successful results
            displayResults(data);

        } catch (error) {
            alert("Error during analysis: " + error.message);
        } finally {
            // 5. Reset UI state
            document.getElementById('loading').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('analyzeBtn').textContent = 'Analyze EEG Signal';
        }
    }

    // Helper to render results to HTML
    function displayResults(data) {
        const resultsDiv = document.getElementById('results');
        const predBox = document.getElementById('predictionBox');
        
        // 1. Basic Info
        document.getElementById('predictionText').textContent = data.Prediction.toUpperCase();
        document.getElementById('confidenceText').textContent = `Confidence: ${(data.Confidence * 100).toFixed(1)}%`;
        predBox.className = 'result-card ' + (data.Prediction === 'Depressed' ? 'depressed' : 'healthy');

        // 2. Clinical Details
        const diagnosisStr = data.Detected_Disorders.length > 0 ? data.Detected_Disorders[0] : 'No Disorder Detected';
        document.getElementById('detailsList').innerHTML = `
            <li><strong>Primary Diagnosis:</strong> ${diagnosisStr}</li>
            <li><strong>Total Windows Analyzed:</strong> ${data.Details.total_windows} (1-second segments)</li>
            <li><strong>Depressive Pattern Windows:</strong> ${data.Details.depressed_windows}</li>
            <li><strong>Healthy Pattern Windows:</strong> ${data.Details.healthy_windows}</li>
        `;

        // 3. Advice / Precautions List
        const adviceListEl = document.getElementById('adviceList');
        const adviceBox = document.getElementById('adviceBox');
        const adviceTitle = document.getElementById('adviceTitle');

        if (adviceListEl && adviceBox && adviceTitle) {
             // Clear previous list
             adviceListEl.innerHTML = "";

             // Populate list from JSON data
             if (data.Advice && data.Advice.length > 0) {
                 data.Advice.forEach(item => {
                     // Simple formatting: bold text between **
                     const formattedItem = item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                     const li = document.createElement('li');
                     li.innerHTML = formattedItem;
                     li.style.marginBottom = "10px";
                     li.style.lineHeight = "1.5";
                     adviceListEl.appendChild(li);
                 });
             }

            // Style the box based on result
            if (data.Prediction === 'Depressed') {
                adviceBox.style.borderLeft = "5px solid #ef4444"; // Red border
                adviceBox.style.backgroundColor = "#fef2f2";
                adviceTitle.innerText = "⚠️ Precautions & Advice";
                adviceTitle.style.color = "#b91c1c";
            } else {
                adviceBox.style.borderLeft = "5px solid #10b981"; // Green border
                adviceBox.style.backgroundColor = "#ecfdf5";
                adviceTitle.innerText = "✨ Well-being Tips";
                adviceTitle.style.color = "#047857";
            }
        }

        // 4. Chart & Show
        renderChart(data.Details.healthy_windows, data.Details.depressed_windows);
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    // Render Chart
    function renderChart(healthy, depressed) {
        const ctx = document.getElementById('probChart').getContext('2d');
        if (probChart) probChart.destroy();

        probChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Healthy Patterns', 'Depressive Patterns'],
                datasets: [{
                    data: [healthy, depressed],
                    backgroundColor: ['#22c55e', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' },
                    title: { display: true, text: 'Signal Classification Distribution', font: { size: 16 } }
                }
            }
        });
    }

    // --- Logout Modal Logic ---
    window.openLogoutModal = function() {
        const modal = document.getElementById('logout-modal');
        const backdrop = document.getElementById('logout-modal-backdrop');
        
        if (modal && backdrop) {
            modal.classList.add('modal-visible');
            backdrop.classList.add('modal-visible');
            
            // Fixes black box bug
            document.body.classList.add('modal-open'); 
            
            setTimeout(() => {
                modal.classList.add('fade-in');
                backdrop.classList.add('fade-in');
            }, 10);
        }
    }

    window.closeLogoutModal = function() {
        const modal = document.getElementById('logout-modal');
        const backdrop = document.getElementById('logout-modal-backdrop');

        if (modal && backdrop) {
            modal.classList.remove('fade-in');
            backdrop.classList.remove('fade-in');
            
            // Restores background
            document.body.classList.remove('modal-open');

            setTimeout(() => {
                modal.classList.remove('modal-visible');
                backdrop.classList.remove('modal-visible');
            }, 200);
        }
    }
});