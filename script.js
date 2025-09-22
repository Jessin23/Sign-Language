// ---------------- SETTINGS ----------------
const MODEL_BASE_PATH = '/models'; // adjust to your server path
const LABELS_BASE_PATH = '/labels'; // adjust to your server path

const SEQUENCE_LENGTH = 30;
const SMOOTHING_WINDOW = 5;
const CONFIDENCE_THRESHOLD = 0.5;
const PREDICTION_INTERVAL = 250; // ms

// ---------------- GLOBALS ----------------
let currentLanguage = 'ASL'; // or 'FSL', set dynamically as needed
let model = null;
let labelMap = null;

const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output');
const canvasCtx = canvasElement.getContext('2d');
const predictionText = document.getElementById('prediction-text');

let hands = null;

let sequence = [];
let predictionsQueue = [];
let lastPredictedClass = null;
let lastPredictionTime = 0;

// ---------------- CAMERA SETUP ----------------
// Access webcam and set video source
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Webcam not supported in this browser.');
    throw new Error('Webcam not supported');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false
  });
  videoElement.srcObject = stream;
  return new Promise(resolve => {
    videoElement.onloadedmetadata = () => resolve(videoElement);
  });
}

// ---------------- LOAD LABEL MAP ----------------
// Load label map JSON and sort keys numerically to ensure correct order
async function loadLabelMap(language) {
  const url = `${LABELS_BASE_PATH}/${language}/label_map.json`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load label map: ${url}`);
  const json = await response.json();
  // Sort keys numerically and map to labels
  const sortedKeys = Object.keys(json).map(Number).sort((a,b) => a-b);
  return sortedKeys.map(k => json[k.toString()]);
}

// ---------------- LOAD MODEL ----------------
// Load TFJS model for the selected language letters
async function loadModel(language) {
  const modelPath = `${MODEL_BASE_PATH}/${language}_letters/model.json`;
  model = await tf.loadLayersModel(modelPath);
  labelMap = await loadLabelMap(language);
  console.log(`✅ Loaded model and labels for ${language} letters`);
}

// ---------------- SWITCH LANGUAGE ----------------
// Switch language, reload model and reset buffers
async function switchLanguage(language) {
  currentLanguage = language;
  await loadModel(language);
  sequence = [];
  predictionsQueue = [];
  lastPredictedClass = null;
  predictionText.textContent = 'No hand detected';
  console.log(`Switched to ${language}`);
}

// ---------------- NORMALIZATION ----------------
// Matches your Python normalization exactly
function normalizeLandmarks(lm) {
  // lm: array of [x,y,z] points length 21
  // Step 1: subtract wrist (lm[0])
  let base = lm[0];
  let normalized = lm.map(p => [p[0] - base[0], p[1] - base[1], p[2] - base[2]]);

  // Step 2: divide by max distance
  let maxDist = 0;
  for (const p of normalized) {
    const dist = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    if (dist > maxDist) maxDist = dist;
  }
  if (maxDist > 0) {
    normalized = normalized.map(p => [p[0]/maxDist, p[1]/maxDist, p[2]/maxDist]);
  }

  // Step 3: special adjustments for thumb_tip (4) and middle_tip (12)
  const thumbTip = normalized[4];
  const indexTip = normalized[8];
  const middleTip = normalized[12];

  normalized[4] = [
    (thumbTip[0] - indexTip[0]) * 1.5,
    (thumbTip[1] - indexTip[1]) * 1.5,
    (thumbTip[2] - indexTip[2]) * 1.5
  ];

  normalized[3] = [
    (normalized[3][0] - middleTip[0]) * 1.2,
    (normalized[3][1] - middleTip[1]) * 1.2,
    (normalized[3][2] - middleTip[2]) * 1.2
  ];

  return normalized;
}

// ---------------- EXTRACT LANDMARKS ----------------
// Extract 21 hand landmarks from MediaPipe results
function extractHandLandmarks(results) {
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return null;
  return results.multiHandLandmarks[0].map(p => [p.x, p.y, p.z]);
}

// ---------------- DRAW LANDMARKS ----------------
// Draw hand landmarks and connections on canvas
function drawLandmarks(results) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.save();
  canvasCtx.scale(-1, 1); // mirror horizontally
  canvasCtx.translate(-canvasElement.width, 0);

  if (results.multiHandLandmarks) {
    for (const lm of results.multiHandLandmarks) {
      window.drawConnectors(canvasCtx, lm, window.HAND_CONNECTIONS, {color:'#00FF00', lineWidth:2});
      window.drawLandmarks(canvasCtx, lm, {color:'#FF0000', lineWidth:1});
    }
  }

  canvasCtx.restore();
}

// ---------------- PREDICTION ----------------
// Predict letters from buffered normalized landmarks sequence
async function predictLetters(results) {
  const landmarks = extractHandLandmarks(results);
  if (!landmarks) {
    sequence = [];
    predictionsQueue = [];
    lastPredictedClass = null;
    predictionText.textContent = 'No hand detected';
    return;
  }

  const normalized = normalizeLandmarks(landmarks);
  sequence.push(normalized.flat());
  if (sequence.length > SEQUENCE_LENGTH) sequence.shift();

  if (sequence.length === SEQUENCE_LENGTH && (Date.now() - lastPredictionTime) > PREDICTION_INTERVAL) {
    const inputTensor = tf.tensor([sequence]);
    const prediction = model.predict(inputTensor);
    const data = await prediction.data();
    inputTensor.dispose();
    prediction.dispose();

    const maxProb = Math.max(...data);
    const maxIndex = data.indexOf(maxProb);
    const predictedClass = maxProb > CONFIDENCE_THRESHOLD ? labelMap[maxIndex] : 'unknown';

    predictionsQueue.push(predictedClass);
    if (predictionsQueue.length > SMOOTHING_WINDOW) predictionsQueue.shift();

    // Majority vote smoothing
    const counts = {};
    for (const p of predictionsQueue) counts[p] = (counts[p] || 0) + 1;
    const sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]);
    const mostCommon = sorted.length > 0 ? sorted[0][0] : 'unknown';

    if (mostCommon !== 'unknown' && mostCommon !== lastPredictedClass) {
      lastPredictedClass = mostCommon;
      predictionText.textContent = `Predicted: ${mostCommon}`;
    } else if (mostCommon === 'unknown') {
      predictionText.textContent = 'Unknown sign';
    }

    lastPredictionTime = Date.now();
  } else if (sequence.length < SEQUENCE_LENGTH) {
    predictionText.textContent = `Collecting frames: ${sequence.length}/${SEQUENCE_LENGTH}`;
  }
}

// ---------------- INITIALIZE MEDIAPIPE HANDS ----------------
// Setup MediaPipe Hands with options and results callback
function initHands() {
  hands = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  hands.onResults(results => {
    drawLandmarks(results);
    predictLetters(results);
  });
}

// ---------------- MAIN LOOP ----------------
// Continuously send video frames to MediaPipe Hands
async function mainLoop() {
  if (videoElement.videoWidth === 0) {
    requestAnimationFrame(mainLoop);
    return;
  }
  if (hands) await hands.send({image: videoElement});
  requestAnimationFrame(mainLoop);
}

// ---------------- MAIN ----------------
// Initialize everything and start detection
async function main() {
  await setupCamera();
  videoElement.play();
  await switchLanguage(currentLanguage); // loads model + label map
  initHands();
  mainLoop();
}

// Start the app
main();
