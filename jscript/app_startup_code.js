//#############################################################

// ### 1. LOAD THE MODEL IMMEDIATELY WHEN THE PAGE LOADS

//#############################################################


// Define 2 helper functions

function simulateClick(tabID) {
	
	document.getElementById(tabID).click();
}



function predictOnLoad() {
	
	// Simulate a click on the predict button
	setTimeout(simulateClick.bind(null,'predict-button'), 500);
}






// LOAD THE MODEL

let model;
(async function () {
	
	// Show loading animation
	$('.progress-bar').show();
	
	try {
		// Try to load the model from local path first
		model = await tf.loadModel('final_model_kaggle_version1/model.json');
		$("#selected-image").attr("src", "assets/samplepic.jpg");
	} catch (e) {
		// Fallback to remote URL if local fails
		console.log("Loading from remote URL due to:", e);
		model = await tf.loadModel('http://skin.test.woza.work/final_model_kaggle_version1/model.json');
		$("#selected-image").attr("src", "http://skin.test.woza.work/assets/samplepic.jpg");
	}
	
	// Hide the model loading spinner
	$('.progress-bar').hide();
	
	// Add a success message
	$('.progress-container').html('<div class="w3-text-green"><i class="fas fa-check-circle"></i> AI model loaded successfully</div>');
	
	// Don't show the sample image on initial load to match the design
	// We'll keep the no-image-selected view visible
	
})();



	

//######################################################################

// ### 2. MAKE A PREDICTION ON THE FRONT PAGE IMAGE WHEN THE PAGE LOADS

//######################################################################



// The model images have size 96x96

// This code is triggered when the predict button is clicked i.e.
// we simulate a click on the predict button.
$("#predict-button").click(async function () {
	
	let image = undefined;
	
	image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224])
	.toFloat();
	
	
	let offset = tf.scalar(127.5);
	
	tensor = tensor.sub(offset)
	.div(offset)
	.expandDims();
	
	
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	
	// TARGET_CLASSES is defined in the target_clssses.js file.
	// There's no need to load this file because it was imported in index.html
	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
	

	// Append the file name to the prediction list
	var file_name = 'Sample Image';
	$("#prediction-list").empty();
	$("#prediction-list").append(`<li class="file-name" style="list-style-type:none;"><i class="fas fa-file-image"></i> ${file_name}</li>`);
	
	top5.forEach(function (p) {
		// Calculate a confidence class based on probability
		let confidenceClass = '';
		let riskLevel = '';
		
		if (p.probability > 0.7) {
			confidenceClass = 'w3-text-red';
			riskLevel = 'High';
		} else if (p.probability > 0.4) {
			confidenceClass = 'w3-text-orange';
			riskLevel = 'Medium';
		} else {
			confidenceClass = 'w3-text-blue';
			riskLevel = 'Low';
		}
		
		// Format the probability as a percentage
		const percentage = (p.probability * 100).toFixed(1) + '%';
		
		// Create a more visually appealing result item
		$("#prediction-list").append(`
			<li class="prediction-item" style="list-style-type:none;">
				<div class="w3-row">
					<div class="w3-col s7">
						<span>${p.className}</span>
					</div>
					<div class="w3-col s3">
						<span class="${confidenceClass}"><b>${percentage}</b></span>
					</div>
					<div class="w3-col s2">
						<span class="${confidenceClass}"><b>${riskLevel}</b></span>
					</div>
				</div>
				<div class="w3-light-grey w3-round-large" style="height:4px; margin-top:5px;">
					<div class="${confidenceClass} w3-round-large" style="height:4px;width:${percentage}"></div>
				</div>
			</li>
		`);
	});
	
});



//######################################################################

// ### 3. READ THE IMAGES THAT THE USER SELECTS

// Then direct the code execution to app_batch_prediction_code.js

//######################################################################




// This listens for a change. It fires when the user submits images.

$("#image-selector").change(async function () {
	
	// the FileReader reads one image at a time
	fileList = $("#image-selector").prop('files');
	
	// Start predicting
	// This function is in the app_batch_prediction_code.js file.
	model_processArray(fileList);
	
});


