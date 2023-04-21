// Run using live-server
// https://stackoverflow.com/a/75418790

async function loadModel() {
  const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  return model;
}

async function predict() {
  const image = document.getElementById('image');
  const model = await loadModel();

  // Preprocess the image.
  const resizedImage = tf.browser.fromPixels(image).toFloat().resizeBilinear([224, 224]);
  const normalizedImage = resizedImage.div(tf.scalar(255));
  const batchedImage = normalizedImage.expandDims(0);

  // Make a prediction.
  const prediction = model.predict(batchedImage);

  // Get the top-1 prediction and display the result.
  const topPrediction = prediction.as1D().argMax().dataSync()[0];
  console.log(prediction)
  document.getElementById('prediction').innerText = `Prediction: ${topPrediction}`;

  // Dispose tensors.
  resizedImage.dispose();
  normalizedImage.dispose();
  batchedImage.dispose();
  prediction.dispose();
}

document.getElementById('predict').addEventListener('click', predict);
