// JavaScript for AgriSmart Application
// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Soil Detection Form Handler
document.getElementById('soilDetectionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('soilImage');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image');
        return;
    }
    
    // Show loading
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('soilResult').style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/detect-soil`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Soil detection failed');
        }
        
        const result = await response.json();
        displaySoilResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error detecting soil. Please try again.');
    } finally {
        document.getElementById('loadingSpinner').style.display = 'none';
    }
});

// Image Preview
document.getElementById('soilImage').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('imagePreview').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// Display Soil Detection Result
function displaySoilResult(result) {
    const resultDiv = document.getElementById('soilResultContent');
    
    const html = `
        <div class="soil-type-card mb-3">
            <h3><i class="fas fa-mountain"></i> ${result.soil_type}</h3>
            <div class="confidence-badge">
                <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <h6><i class="fas fa-chart-bar"></i> Soil Type Probabilities</h6>
                ${Object.entries(result.all_probabilities).map(([type, prob]) => `
                    <div class="mb-2">
                        <div class="d-flex justify-content-between mb-1">
                            <span>${type}</span>
                            <span>${(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${prob * 100}%"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <div class="col-md-6">
                <div class="result-card">
                    <h6><i class="fas fa-info-circle"></i> Soil Properties</h6>
                    <p><strong>Quality Rating:</strong> ${result.quality_rating}</p>
                    <p><strong>Color Features:</strong></p>
                    <ul>
                        <li>Red: ${result.color_features.mean_r.toFixed(2)}</li>
                        <li>Green: ${result.color_features.mean_g.toFixed(2)}</li>
                        <li>Blue: ${result.color_features.mean_b.toFixed(2)}</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    document.getElementById('soilResult').style.display = 'block';
}

// Complete Fertilizer Recommendation Form Handler
document.getElementById('fertilizerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading
    document.getElementById('fertilizerLoading').style.display = 'block';
    document.getElementById('fertilizerResult').style.display = 'none';
    
    // Get form data
    const formData = new FormData();
    formData.append('file', document.getElementById('completeImage').files[0]);
    formData.append('crop', document.getElementById('crop').value);
    formData.append('pH', document.getElementById('pH').value);
    formData.append('nitrogen', document.getElementById('nitrogen').value);
    formData.append('phosphorus', document.getElementById('phosphorus').value);
    formData.append('potassium', document.getElementById('potassium').value);
    formData.append('field_area', document.getElementById('fieldArea').value);
    formData.append('growth_stage', document.getElementById('growthStage').value);
    formData.append('prefer_organic', document.getElementById('preferOrganic').checked);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/complete-recommendation`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Recommendation failed');
        }
        
        const result = await response.json();
        displayFertilizerResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error generating recommendation. Please try again.');
    } finally {
        document.getElementById('fertilizerLoading').style.display = 'none';
    }
});

// Display Fertilizer Recommendation Result
function displayFertilizerResult(data) {
    const resultDiv = document.getElementById('fertilizerResult');
    
    const soilInfo = data.soil_detection;
    const recommendation = data.fertilizer_recommendation;
    
    const html = `
        <div class="alert alert-success">
            <h4 class="alert-heading">
                <i class="fas fa-check-circle"></i> Complete Analysis & Recommendation
            </h4>
        </div>
        
        <!-- Soil Detection Results -->
        <div class="card mb-3">
            <div class="card-header bg-success text-white">
                <h5><i class="fas fa-microscope"></i> Detected Soil Type</h5>
            </div>
            <div class="card-body">
                <h3 class="text-success">${soilInfo.soil_type}</h3>
                <p><strong>Confidence:</strong> ${(soilInfo.confidence * 100).toFixed(2)}%</p>
                <p><strong>Quality:</strong> ${soilInfo.quality_rating}</p>
            </div>
        </div>
        
        <!-- Nutrient Deficit -->
        <div class="card mb-3">
            <div class="card-header bg-warning text-dark">
                <h5><i class="fas fa-exclamation-triangle"></i> Nutrient Deficit Analysis</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <div class="nutrient-box">
                            <h5>Nitrogen</h5>
                            <p>${recommendation.deficit_summary.N_deficit}</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="nutrient-box">
                            <h5>Phosphorus</h5>
                            <p>${recommendation.deficit_summary.P_deficit}</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="nutrient-box">
                            <h5>Potassium</h5>
                            <p>${recommendation.deficit_summary.K_deficit}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Fertilizer Recommendations -->
        <div class="card mb-3">
            <div class="card-header bg-primary text-white">
                <h5><i class="fas fa-flask"></i> Recommended Fertilizers for ${recommendation.crop}</h5>
            </div>
            <div class="card-body">
                ${recommendation.fertilizer_recommendations.map((fert, index) => `
                    <div class="fertilizer-item">
                        <h6>${index + 1}. ${fert.fertilizer}</h6>
                        <p><strong>Quantity:</strong> ${fert.quantity}</p>
                        <p><strong>Nutrients Provided:</strong> ${fert.nutrients}</p>
                        <p><strong>Application:</strong> ${fert.application}</p>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <!-- pH Recommendation -->
        <div class="alert alert-info">
            <h6><i class="fas fa-info-circle"></i> pH Management</h6>
            <p>${recommendation.pH_recommendation}</p>
        </div>
        
        <!-- Soil-Specific Advice -->
        <div class="card mb-3">
            <div class="card-header bg-info text-white">
                <h5><i class="fas fa-lightbulb"></i> Soil-Specific Advice</h5>
            </div>
            <div class="card-body">
                <ul>
                    ${recommendation.soil_specific_advice.map(advice => `
                        <li>${advice}</li>
                    `).join('')}
                </ul>
            </div>
        </div>
        
        <!-- Additional Tips -->
        <div class="tips-section">
            <h5><i class="fas fa-star"></i> Additional Tips</h5>
            <ul>
                ${recommendation.additional_tips.map(tip => `
                    <li>${tip}</li>
                `).join('')}
            </ul>
        </div>
        
        <!-- Timing Advice -->
        <div class="alert alert-secondary mt-3">
            <h6><i class="fas fa-clock"></i> Application Timing</h6>
            <p>${recommendation.timing_advice}</p>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    
    // Scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// Query Processing Form Handler
document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = document.getElementById('queryInput').value;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/process-query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            throw new Error('Query processing failed');
        }
        
        const result = await response.json();
        displayQueryResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing query. Please try again.');
    }
});

// Display Query Result
function displayQueryResult(result) {
    const resultDiv = document.getElementById('queryResult');
    
    const html = `
        <div class="alert alert-info">
            <h5><i class="fas fa-language"></i> Query Analysis</h5>
            <hr>
            <p><strong>Detected Language:</strong> ${result.language_name}</p>
            <p><strong>Intent:</strong> ${result.intent.replace('_', ' ').toUpperCase()}</p>
            
            ${result.entities.crop ? `<p><strong>Detected Crop:</strong> ${result.entities.crop}</p>` : ''}
            ${result.entities.soil_type ? `<p><strong>Detected Soil Type:</strong> ${result.entities.soil_type}</p>` : ''}
            
            ${result.entities.nutrients.length > 0 ? `
                <p><strong>Detected Nutrients:</strong> ${result.entities.nutrients.join(', ')}</p>
            ` : ''}
            
            <div class="alert alert-success mt-3">
                <p class="mb-0">
                    <i class="fas fa-info-circle"></i> 
                    Your query has been analyzed. For specific recommendations, please use the 
                    Fertilizer Recommendation form above with detailed soil parameters.
                </p>
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
}

// Smooth Scrolling for Navigation Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize tooltips (if using Bootstrap tooltips)
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
