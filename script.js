// ===============================
// CONFIG
// ===============================
const MODEL_BASE_PATH = "models";
const LABEL_BASE_PATH = "labels";

let model;
let labelMap = {};
let currentLanguage = "FSL";

let sequence = [];
let predictions = [];
let accumulatedText = "";
let lastTypedChar = "";
const SEQUENCE_LENGTH = 30;
const SMOOTHING_WINDOW = 5;
const CONFIDENCE_THRESHOLD = 0.5;
const PREDICTION_INTERVAL = 250; // ms
const PAUSE_THRESHOLD = 2000; // ms no hand → insert space
let lastPredictionTime = 0;
let lastHandTime = Date.now();

// ===============================
// LOADERS
// ===============================
async function loadLabelMap(language) {
  const res = await fetch(`${LABEL_BASE_PATH}/${language}/label_map.json`);
  if (!res.ok) throw new Error(`Failed to load label map for ${language}`);
  return await res.json();
}

async function loadModel(language) {
  const folder = `${language}_letters`;
  console.log(`Loading model: ${MODEL_BASE_PATH}/${folder}/model.json`);

  model = await tf.loadLayersModel(`${MODEL_BASE_PATH}/${folder}/model.json`);
  labelMap = await loadLabelMap(language);
  console.log(`✅ Loaded ${language} letters model + labels`);
}

// ===============================
// DROPDOWN
// ===============================
document.getElementById("language-dropdown").addEventListener("change", async (e) => {
  currentLanguage = e.target.value;
  sequence = [];
  predictions = [];
  accumulatedText = "";
  lastTypedChar = "";
  await loadModel(currentLanguage);
});

// ===============================
// NORMALIZATION (like Python)
// ===============================
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  let lm = landmarks.map(l => ({
    x: l.x - wrist.x,
    y: l.y - wrist.y,
    z: l.z - wrist.z
  }));

  const maxDist = Math.max(...lm.map(l => Math.sqrt(l.x*l.x + l.y*l.y + l.z*l.z)));
  if (maxDist > 0) lm = lm.map(l => ({ x: l.x/maxDist, y: l.y/maxDist, z: l.z/maxDist }));

  if (lm[4] && lm[8]) {
    lm[4].x = (lm[4].x - lm[8].x) * 1.5;
    lm[4].y = (lm[4].y - lm[8].y) * 1.5;
    lm[4].z = (lm[4].z - lm[8].z) * 1.5;
  }
  if (lm[3] && lm[12]) {
    lm[3].x = (lm[3].x - lm[12].x) * 1.2;
    lm[3].y = (lm[3].y - lm[12].y) * 1.2;
    lm[3].z = (lm[3].z - lm[12].z) * 1.2;
  }

  return lm.flatMap(l => [l.x, l.y, l.z]);
}

// ===============================
// PREDICTION + SMOOTHING
// ===============================
async function predictHand(landmarks) {
  const normalized = normalizeLandmarks(landmarks);
  sequence.push(normalized);
  if (sequence.length > SEQUENCE_LENGTH) sequence.shift();

  if (sequence.length === SEQUENCE_LENGTH) {
    const now = performance.now();
    if (now - lastPredictionTime < PREDICTION_INTERVAL) return null;
    lastPredictionTime = now;

    const input = tf.tensor([sequence]);
    const prediction = model.predict(input);
    const scores = await prediction.data();
    tf.dispose([input, prediction]);

    const maxScore = Math.max(...scores);
    const maxIndex = scores.indexOf(maxScore);

    let currentPrediction = "unknown";
    if (maxScore > CONFIDENCE_THRESHOLD) currentPrediction = labelMap[maxIndex];

    predictions.push(currentPrediction);
    if (predictions.length > SMOOTHING_WINDOW) predictions.shift();

    const counts = {};
    predictions.forEach(p => counts[p] = (counts[p] || 0) + 1);
    const finalPrediction = Object.keys(counts).reduce((a,b)=>counts[a]>counts[b]?a:b);

    return finalPrediction;
  }
  return null;
}

// ===============================
// VIDEO + CANVAS
// ===============================
const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output");
const canvasCtx = canvasElement.getContext("2d");

const hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({ maxNumHands:1, modelComplexity:1, minDetectionConfidence:0.5, minTrackingConfidence:0.5 });

hands.onResults(async results => {
  canvasCtx.save();
  canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
  canvasCtx.scale(-1,1);
  canvasCtx.drawImage(results.image,-canvasElement.width,0,canvasElement.width,canvasElement.height);
  canvasCtx.restore();

  let displayText = "";

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    lastHandTime = Date.now();
    const prediction = await predictHand(results.multiHandLandmarks[0]);

    if (prediction) {
      displayText = `Predicted: ${prediction}`;
      // Only append if new letter or backspace
      if (prediction !== lastTypedChar) {
        if (prediction.toLowerCase() === "backspace" && accumulatedText.length>0) {
          accumulatedText = accumulatedText.slice(0,-1);
        } else if (prediction !== "unknown") {
          accumulatedText += prediction;
        }
        lastTypedChar = prediction;
      }
    } else {
      displayText = `Collecting frames: ${sequence.length}/${SEQUENCE_LENGTH}`;
    }
  } else {
    sequence = [];
    predictions = [];
    lastTypedChar = "";
    displayText = "No hand detected";

    if (Date.now()-lastHandTime>PAUSE_THRESHOLD && accumulatedText && !accumulatedText.endsWith(" ")) {
      accumulatedText += " ";
      lastHandTime = Date.now();
    }
  }

  document.getElementById("prediction-text").textContent = displayText;
  document.getElementById("accumulated-text").textContent = accumulatedText;
});

// ===============================
// CAMERA INIT
// ===============================
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video:true });
  videoElement.srcObject = stream;

  videoElement.onloadedmetadata = () => {
    videoElement.play();
    const sendFrame = async () => {
      await hands.send({ image: videoElement });
      requestAnimationFrame(sendFrame);
    };
    sendFrame();
  };
}

// ===============================
// INITIALIZE APP
// ===============================
(async () => {
  await loadModel(currentLanguage);
  initCamera();
})();
