document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers for info icons
    document.querySelectorAll('.info-icon').forEach(icon => {
        icon.addEventListener('click', function(e) {
            e.preventDefault();
            const formGroup = this.closest('.form-group');
            const input = formGroup.querySelector('input, select');
            const tooltip = formGroup.querySelector('.info-tooltip');
            
            if (input) {
                input.focus();
                formGroup.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Show tooltip
                if (tooltip) {
                    tooltip.style.opacity = '1';
                    tooltip.style.visibility = 'visible';
                    
                    // Hide tooltip after 3 seconds
                    setTimeout(() => {
                        tooltip.style.opacity = '0';
                        tooltip.style.visibility = 'hidden';
                    }, 3000);
                }
            }
        });
    });

    // Hide tooltips when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.info-icon')) {
            document.querySelectorAll('.info-tooltip').forEach(tooltip => {
                tooltip.style.opacity = '0';
                tooltip.style.visibility = 'hidden';
            });
        }
    });

    // Add event listeners to all input fields
    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', function() {
            const infoIcon = this.nextElementSibling;
            if (infoIcon && infoIcon.classList.contains('info-icon')) {
                if (this.value.trim() !== '') {
                    infoIcon.classList.add('visible');
                } else {
                    infoIcon.classList.remove('visible');
                }
            }
        });
    });

    // Existing form submission code
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = {
            age: parseFloat(document.getElementById('age').value),
            gender: parseInt(document.getElementById('gender').value),
            albumin: parseFloat(document.getElementById('albumin').value),
            bilirubin: parseFloat(document.getElementById('bilirubin').value),
            cholesterol: parseFloat(document.getElementById('cholesterol').value),
            copper: parseFloat(document.getElementById('copper').value),
            alk_phos: parseFloat(document.getElementById('alk_phos').value),
            sgot: parseFloat(document.getElementById('sgot').value),
            platelets: parseFloat(document.getElementById('platelets').value),
            prothrombin: parseFloat(document.getElementById('prothrombin').value),
            ascites: parseInt(document.getElementById('ascites').value),
            hepatomegaly: parseInt(document.getElementById('hepatomegaly').value),
            spiders: parseInt(document.getElementById('spiders').value),
            edema: parseInt(document.getElementById('edema').value)
        };

        try {
            // Show loading state
            const submitButton = document.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.textContent = 'Predicting...';

            // Send data to Flask backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            
            // Display result
            const resultDiv = document.getElementById('result');
            const predictionText = document.getElementById('predictionText');
            
            // Calculate risk level based on probability
            const probability = result.probability * 100;
            let riskLevel, riskColor, recommendation;
            
            if (probability < 25) {
                riskLevel = "Low Risk";
                riskColor = "#2ecc71";  // Green
                recommendation = "Your risk of liver cirrhosis is low. Continue maintaining a healthy lifestyle and regular check-ups.";
            } else if (probability < 50 && probability > 25) {
                riskLevel = "Moderate-Low Risk";
                riskColor = "#f1c40f";  // Yellow
                recommendation = "You have a moderate-low risk of liver cirrhosis. Consider lifestyle changes and regular health check-ups.";
            } else if (probability < 75 && probability > 50) {
                riskLevel = "Moderate-High Risk";
                riskColor = "#e67e22";  // Orange
                recommendation = "You have a moderate-high risk of liver cirrhosis. It is recommended to consult with a healthcare professional soon.";
            } else {
                riskLevel = "High Risk";
                riskColor = "#e74c3c";  // Red
                recommendation = "You have a high risk of liver cirrhosis. Please consult with a healthcare professional immediately for proper evaluation and guidance.";
            }
            
            // Create result HTML
            predictionText.innerHTML = `
                <div class="risk-level" style="color: ${riskColor}">
                    <h3>${riskLevel}</h3>
                    <p class="risk-percentage" style="font-size: 1.5rem; font-weight: bold; margin: 1rem 0;">
                        Risk Probability: ${probability.toFixed(1)}%
                    </p>
                </div>
                <div class="recommendation">
                    <p>${recommendation}</p>
                </div>
                <div class="next-steps">
                    <h4>Next Steps:</h4>
                    <ul style="color: ${riskColor}">
                        ${probability >= 50 ? `
                            <li>Schedule an immediate consultation with a hepatologist</li>
                            <li>Get comprehensive liver function tests</li>
                            <li>Stop alcohol consumption immediately</li>
                            <li>Follow a liver-friendly diet</li>
                        ` : `
                            <li>Schedule a routine check-up with your doctor</li>
                            <li>Consider periodic liver function tests</li>
                            <li>Maintain a healthy diet and lifestyle</li>
                            <li>Limit alcohol consumption</li>
                        `}
                    </ul>
                </div>
            `;
            
            resultDiv.style.display = 'block';
            
            // Scroll to result
            resultDiv.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your request. Please try again.');
        } finally {
            // Reset button state
            submitButton.disabled = false;
            submitButton.textContent = 'Predict';
        }
    });
}); 