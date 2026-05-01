// Update this URL to your deployed Render URL once the backend is live
const DEPLOYED_API_URL = 'https://cancer-cell-prediction-api.onrender.com';
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') 
    ? 'http://localhost:8000' 
    : DEPLOYED_API_URL;

// DOM Elements
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const resultCard = document.getElementById('result-card');
const placeholder = document.getElementById('initial-placeholder');
const resultDisplay = document.getElementById('result-display');
const riskCircle = document.getElementById('risk-circle');
const riskScore = document.getElementById('risk-score');
const riskText = document.getElementById('risk-text');
const predictionMsg = document.getElementById('prediction-msg');
const apiStatus = document.getElementById('api-status');
const systemPerf = document.getElementById('system-perf');
const activeModelDisplay = document.getElementById('active-model-display');

// Global state
let loadedModels = [];


// Feature mappings for checkboxes
const checkboxFeatures = [
    'diabetes', 'hypertension', 'asthma', 'cardiac_disease',
    'unexplained_weight_loss', 'persistent_fatigue', 'chronic_pain',
    'abnormal_bleeding', 'persistent_cough', 'lump_presence'
];

/**
 * Initialize App
 */
async function init() {
    await checkApiHealth();
    await loadModels();
}

/**
 * Check Backend Health
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            apiStatus.innerHTML = '<span class="status-indicator status-online"></span> System Online';
            systemPerf.textContent = 'Stable (API v' + data.api_version + ')';
        }
    } catch (error) {
        apiStatus.innerHTML = '<span class="status-indicator status-offline"></span> System Offline';
        systemPerf.textContent = 'Error: Connection Refused';
        console.error('API Health Check Failed:', error);
    }
}

/**
 * Load Available Models
 */
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        loadedModels = await response.json();
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

/**
 * Handle Form Submission
 */
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Loading State
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<div class="spinner"></div> Processing Analysis...';
    
    const formData = new FormData(form);
    const data = {};
    
    // Map regular inputs
    formData.forEach((value, key) => {
        data[key] = isNaN(value) ? value : (value.includes('.') ? parseFloat(value) : parseInt(value));
    });

    // Map checkboxes (boolean to 0/1)
    checkboxFeatures.forEach(feature => {
        data[feature] = document.getElementById(feature).checked ? 1 : 0;
    });

    // Backend still requires tumor_size_cm — provide a default
    if (!data.tumor_size_cm) {
        data.tumor_size_cm = 1.0;
    }

    try {
        // We're always fetching best_model per new UI flow
        const modelName = "best_model"; 
        const response = await fetch(`${API_BASE}/predict?model_name=${modelName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        displayResult(result);

    } catch (error) {
        alert('Error: Could not reach the prediction server. Please ensure the backend is running.');
        console.error('Prediction Failure:', error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Patient Data';
    }
});

/**
 * Display Result in Card
 */
function displayResult(res) {
    placeholder.style.display = 'none';
    resultDisplay.style.display = 'block';
    resultDisplay.className = 'animate-fade';

    const prob = (res.probability * 100).toFixed(1);
    riskScore.textContent = prob + '%';
    riskText.textContent = res.cancer_risk + ' Risk';

    // Update active model display text to show the generic model name vs 'best_model' placeholder
    // backend sends actual model name like 'Gradient Boosting' in the result
    activeModelDisplay.textContent = res.model_used;

    // Remove existing classes
    riskCircle.classList.remove('risk-low', 'risk-medium', 'risk-high');
    
    // Set circle color
    if (res.probability < 0.3) {
        riskCircle.classList.add('risk-low');
        predictionMsg.textContent = 'Routine screening is recommended based on this profile.';
        predictionMsg.style.color = 'var(--success)';
    } else if (res.probability < 0.7) {
        riskCircle.classList.add('risk-medium');
        predictionMsg.textContent = 'Further clinical investigation is advised.';
        predictionMsg.style.color = 'var(--warning)';
    } else {
        riskCircle.classList.add('risk-high');
        predictionMsg.textContent = 'Urgent specialist consultation and diagnostic testing required.';
        predictionMsg.style.color = 'var(--danger)';
    }
}

// ============================================================
// MODE SWITCHING: Upload vs Manual
// ============================================================
const uploadZone = document.getElementById('upload-zone');
const predictionForm = document.getElementById('prediction-form');
const btnUpload = document.getElementById('btn-upload');
const btnManual = document.getElementById('btn-manual');

function switchMode(mode) {
    if (mode === 'upload') {
        uploadZone.style.display = 'block';
        predictionForm.style.display = 'none';
        btnUpload.classList.add('active');
        btnManual.classList.remove('active');
    } else {
        uploadZone.style.display = 'none';
        predictionForm.style.display = 'grid';
        btnUpload.classList.remove('active');
        btnManual.classList.add('active');
    }
}

// Make switchMode globally accessible
window.switchMode = switchMode;

// ============================================================
// DRAG & DROP + FILE INPUT
// ============================================================
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');

['dragenter', 'dragover'].forEach(evt => {
    dropArea.addEventListener(evt, e => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add('drag-over');
    });
});

['dragleave', 'drop'].forEach(evt => {
    dropArea.addEventListener(evt, e => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove('drag-over');
    });
});

dropArea.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

// ============================================================
// FILE PARSING (CSV & JSON)
// ============================================================
function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const reader = new FileReader();

    reader.onload = function(e) {
        try {
            let data;
            if (ext === 'json') {
                const parsed = JSON.parse(e.target.result);
                // Support { key: val } or [{ key: val }]
                data = Array.isArray(parsed) ? parsed[0] : parsed;
            } else if (ext === 'csv') {
                data = parseCSVFirstRow(e.target.result);
            } else {
                showUploadStatus('Unsupported file format. Use CSV or JSON.', false);
                return;
            }

            autoFillForm(data);
            showUploadStatus(`✓ "${file.name}" loaded — ${Object.keys(data).length} fields auto-filled. Review and submit below.`, true);

            // Switch to manual mode so user can review
            switchMode('manual');
        } catch (err) {
            showUploadStatus('Failed to parse file: ' + err.message, false);
        }
    };

    reader.readAsText(file);
}

function parseCSVFirstRow(text) {
    const lines = text.trim().split('\n');
    if (lines.length < 2) throw new Error('CSV must have a header row and at least one data row.');

    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const values = lines[1].split(',').map(v => v.trim().replace(/^"|"$/g, ''));

    const data = {};
    headers.forEach((h, i) => { data[h] = values[i]; });
    return data;
}

// ============================================================
// AUTO-FILL FORM FROM PARSED DATA
// ============================================================
function autoFillForm(data) {
    // Normalize keys: lowercase + underscore
    const normalized = {};
    for (const key in data) {
        normalized[key.toLowerCase().replace(/ /g, '_')] = data[key];
    }

    // Text / number inputs & selects
    const allInputs = predictionForm.querySelectorAll('input[type="number"], input[type="text"], select');
    allInputs.forEach(el => {
        const fieldName = (el.name || el.id || '').toLowerCase();
        if (fieldName && normalized[fieldName] !== undefined) {
            el.value = normalized[fieldName];
        }
    });

    // Checkboxes (boolean / 0-1 fields)
    checkboxFeatures.forEach(feature => {
        const el = document.getElementById(feature);
        if (el && normalized[feature] !== undefined) {
            const val = String(normalized[feature]).trim();
            el.checked = (val === '1' || val.toLowerCase() === 'true' || val.toLowerCase() === 'yes');
        }
    });
}

function showUploadStatus(message, isSuccess) {
    uploadStatus.style.display = 'flex';
    uploadStatus.className = 'upload-status ' + (isSuccess ? 'success' : 'error');
    uploadStatus.textContent = message;
}

// Start health check loop
setInterval(checkApiHealth, 30000);

// Run init
init();
