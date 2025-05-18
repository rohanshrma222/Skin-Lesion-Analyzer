//################################################################################

// ### 1. MAKE A PREDICTION ON THE IMAGE OR MULTIPLE IMAGES THAT THE USER SUBMITS

//#################################################################################





// the model images have size 96x96

async function model_makePrediction(fname) {
	
	//console.log('met_cancer');
	
	// clear the previous variable from memory.
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
	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] // we are selecting the value from the obj
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
		
	// Clear previous predictions
	$("#prediction-list").empty();
	
	// Append the file name to the prediction list
	$("#prediction-list").append(`<li class="file-name" style="list-style-type:none;">
	<i class="fas fa-file-image"></i> ${fname}</li>`);
	
	// Show the results section
	$("#results-section").show();
	
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
	
	// Add a space after the prediction for each image
	$("#prediction-list").append(`<li style="list-style-type:none;"><hr style="margin:15px 0;opacity:0.2;"></li>`);
		
}




// =====================
// The following functions help to solve the problems relating to delays 
// in assigning the src attribute and the delay in model prediction.
// Without this the model will produce unstable predictions because
// it will not be predicting on the correct images.


// This tutorial explains how to use async, await and promises to manage delays.
// Tutorial: https://blog.lavrton.com/javascript-loops-how-to-handle-async-await-6252dd3c795
// =====================



function model_delay() {
	
	return new Promise(resolve => setTimeout(resolve, 200));
}


async function model_delayedLog(item, dataURL) {
	
	// We can await a function that returns a promise.
	// This delays the predictions from appearing.
	// Here it does not actually serve a purpose.
	// It's here to show how a delay like this can be implemented.
	await model_delay();
	
	// display the user submitted image on the page by changing the src attribute.
	$("#selected-image").attr("src", dataURL);
	
	// Show the image preview and hide the placeholder
	$("#no-image-selected").hide();
	$("#image-preview").show();
	
	// log the item only after a delay.
	//console.log(item);
}

// This step by step tutorial explains how to use FileReader.
// Tutorial: http://tutorials.jenkov.com/html5/file-api.html

async function model_processArray(array) {
	
	for(var item of fileList) {
		
		
		let reader = new FileReader();
		
		// clear the previous variable from memory.
		let file = undefined;
	
		
		reader.onload = async function () {
			
			let dataURL = reader.result;
			
			await model_delayedLog(item, dataURL);
			
			
			
			var fname = file.name;
			
			// clear the previous predictions
			$("#prediction-list").empty();
			
			// 'await' is very important here.
			await model_makePrediction(fname);
		}
		
		file = item;
		
		// Print the name of the file to the console
        //console.log("i: " + " - " + file.name);
			
		reader.readAsDataURL(file);
	}
}