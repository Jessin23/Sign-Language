// ===============================
// MEDIA PIPE GLOBAL REFERENCES
// ===============================
window.mp_drawing = window.drawingUtils;
window.mp_hands = window.Hands;
window.mp_holistic = window.Holistic;

console.log('MediaPipe drawing utils:', window.mp_drawing);
console.log('MediaPipe hands:', window.mp_hands);
console.log('MediaPipe holistic:', window.mp_holistic);

// ===============================
// CONFIG & STATE
// ===============================
const MODEL_BASE_PATH = "models";
const LABEL_BASE_PATH = "labels";

let mode = "letters"; // "letters" or "phrases"
let currentLanguage = "FSL";

let model;
let labelMap = {};
let accumulatedText = "";
let lastTypedChar = "";
let lastPhraseWord = "";

// Letters
let sequence = [];
let predictions = [];
const SEQUENCE_LENGTH_LETTERS = 30;
const SMOOTHING_WINDOW_LETTERS = 5;
const CONFIDENCE_THRESHOLD_LETTERS = 0.5;
const PREDICTION_INTERVAL = 250;
let lastPredictionTime = 0;
let lastHandTime = Date.now();

// Phrases
const PHRASE_ACTIONS = {
  FSL: ['kumusta', 'salamat', 'mahal kita', 'none', 'paalam', 'oo', 'patawad'],
  ASL: ['hello', 'thank you', 'I love you', 'none', 'goodbye', 'yes', 'sorry']
};
let phraseSequence = [];
let phrasePredictionsProb = [];
let currentClass = null;
let switchCount = 0;
const SEQUENCE_LENGTH_PHRASES = 30;
const SMOOTHING_WINDOW_PHRASES = 10;
const REQUIRED_CONSISTENCY = 8;
const PHRASE_THRESHOLD = 0.5;

// ===============================
// LOAD MODEL & LABELS
// ===============================
async function loadLabelMap(language) {
  const res = await fetch(`${LABEL_BASE_PATH}/${language}/label_map.json`);
  if (!res.ok) throw new Error(`Failed to load label map for ${language}`);
  const rawLabelMap = await res.json();
  if (Array.isArray(rawLabelMap)) return rawLabelMap;
  return Object.keys(rawLabelMap).sort((a, b) => parseInt(a) - parseInt(b)).map(k => rawLabelMap[k]);
}

async function loadModel(language, mode) {
  let folder = mode === "letters" ? `${language}_letters` : `${language}_phrases`;
  model = await tf.loadLayersModel(`${MODEL_BASE_PATH}/${folder}/model.json`);
  labelMap = mode === "letters" ? await loadLabelMap(language) : PHRASE_ACTIONS[language];

  try {
    await tf.setBackend('webgl');
    await tf.ready();
  } catch (e) { console.warn(e); }

  // warm up
  const featureSize = mode === 'letters' ? 21 * 3 : (33 * 4) + (468 * 3) + (21 * 3) + (21 * 3);
  const dummy = tf.zeros([1, mode === 'letters' ? SEQUENCE_LENGTH_LETTERS : SEQUENCE_LENGTH_PHRASES, featureSize]);
  const out = model.predict(dummy); if (out && out.dataSync) out.dataSync();
  tf.dispose([dummy, out]);
}

// ===============================
// UI HANDLERS
// ===============================
document.getElementById("btn-letters").addEventListener("click", async () => {
  if (mode === "letters") return;
  mode = "letters";
  document.getElementById("btn-letters").classList.add("active");
  document.getElementById("btn-phrases").classList.remove("active");
  await loadModel(currentLanguage, mode);
  resetSequences();
  updateUI("Waiting for detection...");
});

document.getElementById("btn-phrases").addEventListener("click", async () => {
  if (mode === "phrases") return;
  mode = "phrases";
  document.getElementById("btn-phrases").classList.add("active");
  document.getElementById("btn-letters").classList.remove("active");
  await loadModel(currentLanguage, mode);
  resetSequences();
  updateUI("Waiting for detection...");
});

document.getElementById("language-dropdown").addEventListener("change", async (e) => {
  currentLanguage = e.target.value;
  await loadModel(currentLanguage, mode);
  resetSequences();
  updateUI("Waiting for detection...");
});

document.getElementById("clear-btn").addEventListener("click", () => {
  resetSequences();
  updateUI("Waiting for detection...");
});

function resetSequences() {
  sequence = []; predictions = [];
  phraseSequence = []; phrasePredictionsProb = [];
  accumulatedText = ""; lastTypedChar = ""; currentClass = null; lastPhraseWord = "";
  lastPredictionTime = 0; lastHandTime = Date.now();
}

// ===============================
// NORMALIZATION
// ===============================
function normalizeHandLandmarks(landmarks) {
  let flatLm = landmarks.flatMap(l => [l.x, l.y, l.z]);
  const wristX = flatLm[0], wristY = flatLm[1], wristZ = flatLm[2];
  let centered = [];
  for (let i = 0; i < flatLm.length; i += 3) {
    centered.push(flatLm[i] - wristX);
    centered.push(flatLm[i + 1] - wristY);
    centered.push(flatLm[i + 2] - wristZ);
  }

  const maxDist = Math.max(...centered.map((v, i) => i % 3 === 0 ? Math.sqrt(centered[i] ** 2 + centered[i + 1] ** 2 + centered[i + 2] ** 2) : 0).filter(d => d > 0));
  let scaled = centered.map(v => maxDist > 0 ? v / maxDist : 0);

  // Custom thumb transformations
  if (scaled.length >= (12 * 3 + 3)) {
    const thumb_tip_x = scaled[4 * 3], thumb_tip_y = scaled[4 * 3 + 1], thumb_tip_z = scaled[4 * 3 + 2];
    const index_tip_x = scaled[8 * 3], index_tip_y = scaled[8 * 3 + 1], index_tip_z = scaled[8 * 3 + 2];
    const middle_tip_x = scaled[12 * 3], middle_tip_y = scaled[12 * 3 + 1], middle_tip_z = scaled[12 * 3 + 2];
    const lm3_x = scaled[3 * 3], lm3_y = scaled[3 * 3 + 1], lm3_z = scaled[3 * 3 + 2];

    scaled[4 * 3] = (thumb_tip_x - index_tip_x) * 1.5;
    scaled[4 * 3 + 1] = (thumb_tip_y - index_tip_y) * 1.5;
    scaled[4 * 3 + 2] = (thumb_tip_z - index_tip_z) * 1.5;

    scaled[3 * 3] = (lm3_x - middle_tip_x) * 1.2;
    scaled[3 * 3 + 1] = (lm3_y - middle_tip_y) * 1.2;
    scaled[3 * 3 + 2] = (lm3_z - middle_tip_z) * 1.2;
  }

  return scaled;
}

function normalizeHolisticKeypoints(keypoints) {
  if (!keypoints || keypoints.length === 0) return null;
  let normalized = [...keypoints];
  const torsoX = normalized[0], torsoY = normalized[1];

  const POSE_START = 0, POSE_LENGTH = 33 * 4;
  const FACE_START = POSE_LENGTH, FACE_LENGTH = 468 * 3;
  const LH_START = FACE_START + FACE_LENGTH, LH_LENGTH = 21 * 3;
  const RH_START = LH_START + LH_LENGTH, RH_LENGTH = 21 * 3;

  for (let i = POSE_START; i < POSE_START + POSE_LENGTH && i < normalized.length; i += 4) {
    normalized[i] -= torsoX; normalized[i + 1] -= torsoY;
  }
  for (let i = FACE_START; i < FACE_START + FACE_LENGTH && i < normalized.length; i += 3) {
    normalized[i] -= torsoX; normalized[i + 1] -= torsoY;
  }
  for (let i = LH_START; i < LH_START + LH_LENGTH && i < normalized.length; i += 3) {
    normalized[i] -= torsoX; normalized[i + 1] -= torsoY;
  }
  for (let i = RH_START; i < RH_START + RH_LENGTH && i < normalized.length; i += 3) {
    normalized[i] -= torsoX; normalized[i + 1] -= torsoY;
  }

  const maxVal = Math.max(...normalized.map(Math.abs));
  return normalized.map(v => maxVal > 0 ? v / maxVal : 0);
}

// ===============================
// CAMERA + MEDIA PIPE
// ===============================
const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output");
const canvasCtx = canvasElement.getContext("2d");

const hands = new Hands({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
hands.onResults(onHandsResults);

const holistic = new Holistic({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
holistic.setOptions({ modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
holistic.onResults(onHolisticResults);

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;

      const sendFrame = async () => {
        if (!videoElement.paused && !videoElement.ended) {
          if (mode === "letters") await hands.send({ image: videoElement });
          else await holistic.send({ image: videoElement });
        }
        requestAnimationFrame(sendFrame);
      };
      sendFrame();
    };
  } catch (error) {
    console.error('Error accessing camera:', error);
    updateUI('Camera access denied. Please allow camera permissions.');
  }
}

// ===============================
// PREDICTION FUNCTIONS
// ===============================
async function predictLetter(landmarks) {
  const normalized = normalizeHandLandmarks(landmarks);
  sequence.push(normalized);
  if (sequence.length > SEQUENCE_LENGTH_LETTERS) sequence.shift();

  if (sequence.length === SEQUENCE_LENGTH_LETTERS) {
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
    if (maxScore >= CONFIDENCE_THRESHOLD_LETTERS) currentPrediction = labelMap[maxIndex];

    predictions.push(currentPrediction);
    if (predictions.length > SMOOTHING_WINDOW_LETTERS) predictions.shift();

    const counts = {};
    predictions.forEach(p => counts[p] = (counts[p] || 0) + 1);
    const sortedCounts = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const mostCommon = sortedCounts.length > 0 ? sortedCounts[0][0] : "unknown";
    const mostCommonCount = sortedCounts.length > 0 ? sortedCounts[0][1] : 0;

    if (mostCommon !== "unknown" && mostCommonCount >= Math.floor(SMOOTHING_WINDOW_LETTERS / 2)) return mostCommon;
    return "unknown";
  }
  return null;
}

async function predictPhrase(keypoints) {
  if (!keypoints) {
    phraseSequence = []; phrasePredictionsProb = [];
    if (currentClass !== 'none') switchCount = 0;
    currentClass = 'none';
    return 'none';
  }

  phraseSequence.push(keypoints);
  if (phraseSequence.length > SEQUENCE_LENGTH_PHRASES) phraseSequence.shift();

  let predictedClass = currentClass;
  if (phraseSequence.length === SEQUENCE_LENGTH_PHRASES) {
    const now = performance.now();
    if (now - lastPredictionTime < PREDICTION_INTERVAL) return predictedClass;
    lastPredictionTime = now;

    const input = tf.tensor([phraseSequence]);
    const res = await model.predict(input).data();
    tf.dispose(input);

    phrasePredictionsProb.push(res);
    if (phrasePredictionsProb.length > SMOOTHING_WINDOW_PHRASES) phrasePredictionsProb.shift();

    const avgRes = Array(res.length).fill(0);
    phrasePredictionsProb.forEach(arr => arr.forEach((v, i) => avgRes[i] += v));
    for (let i = 0; i < avgRes.length; i++) avgRes[i] /= phrasePredictionsProb.length;

    const maxIdx = avgRes.indexOf(Math.max(...avgRes));

    if (avgRes[maxIdx] >= PHRASE_THRESHOLD) {
      if (currentClass !== labelMap[maxIdx]) {
        switchCount++;
        if (switchCount >= REQUIRED_CONSISTENCY) {
          currentClass = labelMap[maxIdx];
          switchCount = 0;
        }
      } else switchCount = 0;
    } else { currentClass = "none"; switchCount = 0; }
    predictedClass = currentClass;
  }
  return predictedClass;
}

// ===============================
// UI UPDATE
// ===============================
function updateUI(predicted) {
  document.getElementById("prediction-text").textContent = `Predicted: ${predicted}`;
  document.getElementById("accumulated-text").textContent = accumulatedText;
}

// ===============================
// HANDS & HOLISTIC RESULTS
// ===============================
async function onHandsResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(results.image, -canvasElement.width, 0, canvasElement.width, canvasElement.height);
  canvasCtx.restore();

  let displayText = "No hand detected";
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    lastHandTime = Date.now();
    if (window.mp_drawing && window.mp_hands) {
      window.mp_drawing.drawConnectors(canvasCtx, results.multiHandLandmarks[0], window.mp_hands.HAND_CONNECTIONS,
        { color: '#00FF00', lineWidth: 5 });
      window.mp_drawing.drawLandmarks(canvasCtx, results.multiHandLandmarks[0], { color: '#FF0000', lineWidth: 2 });
    }

    const prediction = await predictLetter(results.multiHandLandmarks[0]);
    if (prediction) {
      displayText = `Predicted: ${prediction}`;
      if (prediction !== "unknown" && prediction !== lastTypedChar) {
        if (prediction.toLowerCase() === "backspace" && accumulatedText.length > 0) accumulatedText = accumulatedText.slice(0, -1);
        else accumulatedText += prediction;
        lastTypedChar = prediction;
      }
    } else displayText = `Collecting frames: ${sequence.length}/${SEQUENCE_LENGTH_LETTERS}`;
  } else {
    const PAUSE_THRESHOLD_MS = 2000;
    if (Date.now() - lastHandTime > PAUSE_THRESHOLD_MS && accumulatedText && !accumulatedText.endsWith(" ")) {
      accumulatedText += " "; lastHandTime = Date.now();
    }
    sequence = []; predictions = []; lastTypedChar = "";
  }
  updateUI(displayText);
}

async function onHolisticResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(results.image, -canvasElement.width, 0, canvasElement.width, canvasElement.height);
  canvasCtx.restore();

  if (window.mp_drawing && window.mp_holistic) {
    if (results.faceLandmarks) window.mp_drawing.drawConnectors(canvasCtx, results.faceLandmarks, window.mp_holistic.FACEMESH_TESSELATION, {color: '#808080', lineWidth:1});
    if (results.poseLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.poseLandmarks, window.mp_holistic.POSE_CONNECTIONS, {color:'#00FF00', lineWidth:2});
      window.mp_drawing.drawLandmarks(canvasCtx, results.poseLandmarks, {color:'#FF0000', lineWidth:5});
    }
    if (results.leftHandLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.leftHandLandmarks, window.mp_holistic.HAND_CONNECTIONS, {color:'#00FFFF', lineWidth:2});
      window.mp_drawing.drawLandmarks(canvasCtx, results.leftHandLandmarks, {color:'#FF00FF', lineWidth:5});
    }
    if (results.rightHandLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.rightHandLandmarks, window.mp_holistic.HAND_CONNECTIONS, {color:'#FFFF00', lineWidth:2});
      window.mp_drawing.drawLandmarks(canvasCtx, results.rightHandLandmarks, {color:'#00FF00', lineWidth:5});
    }
  }

  const keypoints = [];
  const hasPose = !!results.poseLandmarks;
  const hasFace = !!results.faceLandmarks;
  const hasLH = !!results.leftHandLandmarks;
  const hasRH = !!results.rightHandLandmarks;
  const anyDetected = hasPose || hasFace || hasLH || hasRH;

  if (hasPose) results.poseLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z, l.visibility));
  else keypoints.push(...Array(33*4).fill(0));
  if (hasFace) results.faceLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z));
  else keypoints.push(...Array(468*3).fill(0));
  if (hasLH) results.leftHandLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z));
  else keypoints.push(...Array(21*3).fill(0));
  if (hasRH) results.rightHandLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z));
  else keypoints.push(...Array(21*3).fill(0));

  const normalized = normalizeHolisticKeypoints(keypoints);
  const predicted = await predictPhrase(normalized);
  updateUI(predicted);
}

// ===============================
// START
// ===============================
(async () => {
  await loadModel(currentLanguage, mode);
  await initCamera();
})();
