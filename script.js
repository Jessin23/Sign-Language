// ===============================
// MEDIAPIPE GLOBAL REFERENCES
// ===============================
// Ensure MediaPipe objects are available globally
window.mp_drawing = window.drawingUtils;
window.mp_hands = window.Hands;
window.mp_holistic = window.Holistic;

// Debug check
console.log('MediaPipe drawing utils:', window.mp_drawing);
console.log('MediaPipe hands:', window.mp_hands);
console.log('MediaPipe holistic:', window.mp_holistic);

// ===============================
// CONFIG
// ===============================
const MODEL_BASE_PATH = "models";
const LABEL_BASE_PATH = "labels";

let mode = "letters"; // "letters" or "phrases"
let currentLanguage = "FSL";

let model;
let labelMap = {};
let accumulatedText = "";
let lastTypedChar = "";
let lastPhraseWord = ""; // Added for phrase mode to prevent spamming

// Letters settings
let sequence = [];
let predictions = [];
const SEQUENCE_LENGTH_LETTERS = 30;
const SMOOTHING_WINDOW_LETTERS = 5;
const CONFIDENCE_THRESHOLD_LETTERS = 0.5;
const PREDICTION_INTERVAL = 250; // milliseconds
let lastPredictionTime = 0;
let lastHandTime = Date.now(); // For auto-space

// Phrases settings
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
// LOADERS
// ===============================
async function loadLabelMap(language) {
  const res = await fetch(`${LABEL_BASE_PATH}/${language}/label_map.json`);
  if (!res.ok) throw new Error(`Failed to load label map for ${language}`);
  // Assuming label_map.json is an object like {"0": "A", "1": "B"}
  // or an array directly. If it's an object, we need to sort it by keys.
  const rawLabelMap = await res.json();
  if (Array.isArray(rawLabelMap)) {
      return rawLabelMap; // Already an array, assume correct order
  } else {
      // If it's an object, sort by numeric keys to match Python's approach
      return Object.keys(rawLabelMap)
                   .sort((a, b) => parseInt(a) - parseInt(b))
                   .map(key => rawLabelMap[key]);
  }
}

async function loadModel(language, mode) {
  let folder;
  if (mode === "letters") {
    folder = `${language}_letters`;
    console.log(`Loading letters model: ${MODEL_BASE_PATH}/${folder}/model.json`);
    model = await tf.loadLayersModel(`${MODEL_BASE_PATH}/${folder}/model.json`);
    labelMap = await loadLabelMap(language);
  } else { // phrases mode
    folder = `${language}_phrases`;
    console.log(`Loading phrases model: ${MODEL_BASE_PATH}/${folder}/model.json`);
    model = await tf.loadLayersModel(`${MODEL_BASE_PATH}/${folder}/model.json`);
    // For phrases, labelMap comes directly from PHRASE_ACTIONS array
    labelMap = PHRASE_ACTIONS[language];
  }

  // Set TFJS backend and warm up for stability
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('TFJS backend set to WebGL and ready.');
  } catch (e) {
    console.warn('TFJS backend set failed, using default (CPU):', e);
  }

  try {
    // Determine feature size dynamically based on landmark structure
    let featureSize;
    if (mode === 'letters') {
      featureSize = 21 * 3; // 21 landmarks * (x,y,z)
    } else {
      // Pose (33*4) + Face (468*3) + LH (21*3) + RH (21*3)
      featureSize = (33 * 4) + (468 * 3) + (21 * 3) + (21 * 3);
    }
    const dummy = tf.zeros([1, mode === 'letters' ? SEQUENCE_LENGTH_LETTERS : SEQUENCE_LENGTH_PHRASES, featureSize]);
    const out = model.predict(dummy);
    if (out && out.dataSync) out.dataSync();
    tf.dispose([dummy, out]);
    console.log('Model warmed up successfully.');
  } catch (e) {
    console.warn('Warmup inference failed:', e);
  }
}

// ===============================
// UI HANDLERS
// ===============================
document.getElementById("btn-letters").addEventListener("click", async () => {
  if (mode === "letters") return; // Prevent unnecessary reloads
  mode = "letters";
  document.getElementById("btn-letters").classList.add("active");
  document.getElementById("btn-phrases").classList.remove("active");
  await loadModel(currentLanguage, mode);
  resetSequences();
  updateUI("Waiting for detection...");
});

document.getElementById("btn-phrases").addEventListener("click", async () => {
  if (mode === "phrases") return; // Prevent unnecessary reloads
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
  resetSequences(); // Clear all text and prediction states
  updateUI("Waiting for detection...");
});

function resetSequences() {
  sequence = [];
  predictions = [];
  phraseSequence = [];
  phrasePredictionsProb = [];
  accumulatedText = "";
  lastTypedChar = "";
  currentClass = null; // Reset current class for phrases
  lastPhraseWord = ""; // Reset last phrase word
  lastPredictionTime = 0; // Reset prediction timing
  lastHandTime = Date.now(); // Reset hand detection time for auto-space
}

// ===============================
// NORMALIZATION FUNCTION FOR LETTERS (FIXED)
// ===============================
function normalizeHandLandmarks(landmarks) {
  // Landmarks come in as an array of objects {x,y,z}.
  // We need to convert them to a flat array for consistent arithmetic.
  let flatLm = landmarks.flatMap(l => [l.x, l.y, l.z]); // Flatten original for wrist centering

  // Python's `lm -= lm[0]` means subtracting wrist_x, wrist_y, wrist_z from all points.
  // Then the result is re-flattened implicitly by numpy.
  const wristX = flatLm[0];
  const wristY = flatLm[1];
  const wristZ = flatLm[2];

  let centeredLm = [];
  for (let i = 0; i < flatLm.length; i += 3) {
      centeredLm.push(flatLm[i] - wristX);     // x
      centeredLm.push(flatLm[i + 1] - wristY); // y
      centeredLm.push(flatLm[i + 2] - wristZ); // z
  }

  // Scale to max distance
  const maxDist = Math.max(...centeredLm.map((_, i) =>
      (i % 3 === 0) ? Math.sqrt(centeredLm[i] * centeredLm[i] + centeredLm[i + 1] * centeredLm[i + 1] + centeredLm[i + 2] * centeredLm[i + 2]) : 0
  ).filter(d => d > 0)); // Filter out zeros from non-x coordinates

  let scaledLm = centeredLm.map(v => (maxDist > 0 ? v / maxDist : 0));

  // --- Crucial Custom Transformations for Thumb (FIXED) ---
  // Apply the same custom transformations as in Python.
  // These operations are on the _scaled_ landmark coordinates.
  // lm[4] is thumb_tip (indices 12,13,14 in scaledLm)
  // lm[8] is index_tip (indices 24,25,26 in scaledLm)
  // lm[12] is middle_tip (indices 36,37,38 in scaledLm)
  // lm[3] is thumb_ip (indices 9,10,11 in scaledLm)

  if (scaledLm.length >= (12*3 + 3)) { // Ensure enough landmarks for these operations
      // Calculate derived points from existing scaled landmarks
      const thumb_tip_x = scaledLm[4 * 3];     const thumb_tip_y = scaledLm[4 * 3 + 1];     const thumb_tip_z = scaledLm[4 * 3 + 2];
      const index_tip_x = scaledLm[8 * 3];     const index_tip_y = scaledLm[8 * 3 + 1];     const index_tip_z = scaledLm[8 * 3 + 2];
      const middle_tip_x = scaledLm[12 * 3];   const middle_tip_y = scaledLm[12 * 3 + 1];   const middle_tip_z = scaledLm[12 * 3 + 2];
      const lm3_x = scaledLm[3 * 3];           const lm3_y = scaledLm[3 * 3 + 1];           const lm3_z = scaledLm[3 * 3 + 2];

      // Apply transformations to the specific landmark indices in the scaledLm array
      // For lm[4] (thumb tip)
      scaledLm[4 * 3]     = (thumb_tip_x - index_tip_x) * 1.5;
      scaledLm[4 * 3 + 1] = (thumb_tip_y - index_tip_y) * 1.5;
      scaledLm[4 * 3 + 2] = (thumb_tip_z - index_tip_z) * 1.5;

      // For lm[3] (thumb IP joint)
      scaledLm[3 * 3]     = (lm3_x - middle_tip_x) * 1.2;
      scaledLm[3 * 3 + 1] = (lm3_y - middle_tip_y) * 1.2;
      scaledLm[3 * 3 + 2] = (lm3_z - middle_tip_z) * 1.2;
  }
  // --- END Custom Transformations ---

  return scaledLm;
}

// ===============================
// NORMALIZATION FUNCTION FOR PHRASES (FIXED)
// ===============================
function normalizeHolisticKeypoints(keypoints) {
    if (!keypoints || keypoints.length === 0) return null;

    // Make a mutable copy of the keypoints array
    let normalized = [...keypoints];

    // Python uses keypoints[0] (nose x) and keypoints[1] (nose y) to center all other points.
    // The Python `keypoints[0::4]` and `keypoints[1::4]` slices target ALL x and y coordinates
    // regardless of the landmark type, relative to the start of the array.
    // We need to replicate that by iterating through the different parts.

    // Assume the first 4 elements are nose_x, nose_y, nose_z, nose_visibility
    const torsoX = normalized[0];
    const torsoY = normalized[1];

    // Offsets for each landmark type's data within the flattened array
    const POSE_START = 0;
    const POSE_LENGTH = 33 * 4; // x,y,z,visibility for 33 pose landmarks
    const FACE_START = POSE_LENGTH;
    const FACE_LENGTH = 468 * 3; // x,y,z for 468 face landmarks
    const LH_START = FACE_START + FACE_LENGTH;
    const LH_LENGTH = 21 * 3;    // x,y,z for 21 left hand landmarks
    const RH_START = LH_START + LH_LENGTH;
    const RH_LENGTH = 21 * 3;    // x,y,z for 21 right hand landmarks

    // Centering Pose Landmarks (x,y,z,visibility)
    for (let i = POSE_START; i < POSE_START + POSE_LENGTH && i < normalized.length; i += 4) {
        normalized[i] -= torsoX;     // X coordinate
        normalized[i + 1] -= torsoY; // Y coordinate
    }

    // Centering Face Landmarks (x,y,z)
    for (let i = FACE_START; i < FACE_START + FACE_LENGTH && i < normalized.length; i += 3) {
        normalized[i] -= torsoX;     // X coordinate
        normalized[i + 1] -= torsoY; // Y coordinate
    }

    // Centering Left Hand Landmarks (x,y,z)
    for (let i = LH_START; i < LH_START + LH_LENGTH && i < normalized.length; i += 3) {
        normalized[i] -= torsoX;     // X coordinate
        normalized[i + 1] -= torsoY; // Y coordinate
    }

    // Centering Right Hand Landmarks (x,y,z)
    for (let i = RH_START; i < RH_START + RH_LENGTH && i < normalized.length; i += 3) {
        normalized[i] -= torsoX;     // X coordinate
        normalized[i + 1] -= torsoY; // Y coordinate
    }

    // Scale by the maximum absolute value
    const maxVal = Math.max(...normalized.map(Math.abs));
    return normalized.map(v => maxVal > 0 ? v / maxVal : 0);
}

// ===============================
// LETTERS LIVE DETECTION
// ===============================
async function predictLetter(landmarks) {
  const normalized = normalizeHandLandmarks(landmarks);
  sequence.push(normalized);
  if (sequence.length > SEQUENCE_LENGTH_LETTERS) sequence.shift();

  if (sequence.length === SEQUENCE_LENGTH_LETTERS) {
    const now = performance.now();
    if (now - lastPredictionTime < PREDICTION_INTERVAL) return null; // Use current prediction if too soon
    lastPredictionTime = now;

    const input = tf.tensor([sequence]); // Shape: [1, SEQUENCE_LENGTH, featureSize]
    const prediction = model.predict(input);
    const scores = await prediction.data(); // Get raw probabilities
    tf.dispose([input, prediction]);

    const maxScore = Math.max(...scores);
    const maxIndex = scores.indexOf(maxScore);
    let currentPrediction = "unknown";
    if (maxScore >= CONFIDENCE_THRESHOLD_LETTERS) { // Use >= for consistency with Python
        currentPrediction = labelMap[maxIndex];
    }

    predictions.push(currentPrediction);
    if (predictions.length > SMOOTHING_WINDOW_LETTERS) predictions.shift();

    // Majority vote smoothing
    const counts = {};
    predictions.forEach(p => counts[p] = (counts[p] || 0) + 1);

    // Python uses `most_common(1)[0]` which returns (item, count).
    // It also checks `count >= (smoothing_window // 2)`.
    const sortedCounts = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const mostCommon = sortedCounts.length > 0 ? sortedCounts[0][0] : "unknown";
    const mostCommonCount = sortedCounts.length > 0 ? sortedCounts[0][1] : 0;

    if (mostCommon !== "unknown" && mostCommonCount >= Math.floor(SMOOTHING_WINDOW_LETTERS / 2)) {
        return mostCommon;
    }
    return "unknown"; // Return unknown if not confident after smoothing
  }
  return null; // Sequence not full yet
}

// ===============================
// PREDICTION & SMOOTHING FOR PHRASES
// ===============================
async function predictPhrase(keypoints) {
  if (!keypoints) {
    // If no keypoints, act as if "none" was predicted with 100% confidence
    phraseSequence = [];
    phrasePredictionsProb = [];
    // Directly set currentClass to 'none' and reset switch count
    if (currentClass !== 'none') { // Only increment if switching from something else to 'none'
        switchCount = 0; // Reset switch count if no keypoints
    }
    currentClass = 'none';
    return 'none';
  }

  phraseSequence.push(keypoints);
  if (phraseSequence.length > SEQUENCE_LENGTH_PHRASES) phraseSequence.shift();

  let predictedClass = currentClass; // Default to current_class if no new prediction
  if (phraseSequence.length === SEQUENCE_LENGTH_PHRASES) {
    const now = performance.now();
    if (now - lastPredictionTime < PREDICTION_INTERVAL) {
        return predictedClass; // Return current stabilized class if too soon
    }
    lastPredictionTime = now;

    const input = tf.tensor([phraseSequence]);
    const res = await model.predict(input).data(); // Get raw probabilities
    tf.dispose(input);

    phrasePredictionsProb.push(res);
    if (phrasePredictionsProb.length > SMOOTHING_WINDOW_PHRASES) phrasePredictionsProb.shift();

    // Average probabilities for smoothing
    const avgRes = Array(res.length).fill(0);
    phrasePredictionsProb.forEach(arr => arr.forEach((v, i) => avgRes[i] += v));
    for (let i = 0; i < avgRes.length; i++) avgRes[i] /= phrasePredictionsProb.length;

    const maxIdx = avgRes.indexOf(Math.max(...avgRes));

    // Class switching logic
    if (avgRes[maxIdx] >= PHRASE_THRESHOLD) { // Use >= for consistency with Python
      if (currentClass !== labelMap[maxIdx]) {
        switchCount++;
        if (switchCount >= REQUIRED_CONSISTENCY) {
          currentClass = labelMap[maxIdx];
          switchCount = 0; // Reset switch count after successful switch
        }
      } else {
        switchCount = 0; // Reset if consistency is maintained
      }
    } else {
      currentClass = "none"; // If below threshold, consider it 'none'
      switchCount = 0; // Reset switch count
    }
    predictedClass = currentClass; // Update predictedClass after logic
  }
  return predictedClass;
}

// ===============================
// UI UPDATES
// ===============================
function updateUI(predicted) {
  document.getElementById("prediction-text").textContent = `Predicted: ${predicted}`;
  document.getElementById("accumulated-text").textContent = accumulatedText;
}

// ===============================
// CAMERA + MEDIA PIPE
// ===============================
const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output");
const canvasCtx = canvasElement.getContext("2d");

// Setup for Hands (Letters mode)
const hands = new Hands({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
hands.onResults(onHandsResults); // Link results to a dedicated handler

// Setup for Holistic (Phrases mode)
const holistic = new Holistic({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
holistic.setOptions({ modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
holistic.onResults(onHolisticResults); // Link results to a dedicated handler

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;

      const sendFrame = async () => {
        // Only send if video is actively playing to avoid errors
        if (!videoElement.paused && !videoElement.ended) {
          if (mode === "letters") {
            await hands.send({ image: videoElement });
          } else { // phrases mode
            await holistic.send({ image: videoElement });
          }
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
// HANDS (LETTERS) RESULTS HANDLER
// ===============================
async function onHandsResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.scale(-1, 1); // Flip horizontally for selfie-view
  canvasCtx.drawImage(results.image, -canvasElement.width, 0, canvasElement.width, canvasElement.height);
  canvasCtx.restore();

  let displayText = "No hand detected";
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    lastHandTime = Date.now(); // Reset pause timer
    // Draw landmarks - USING WINDOW REFERENCES FOR SAFETY
    if (window.mp_drawing && window.mp_hands) {
      window.mp_drawing.drawConnectors(canvasCtx, results.multiHandLandmarks[0], window.mp_hands.HAND_CONNECTIONS,
                                     { color: '#00FF00', lineWidth: 5 });
      window.mp_drawing.drawLandmarks(canvasCtx, results.multiHandLandmarks[0], 
                                    { color: '#FF0000', lineWidth: 2 });
    }

    const prediction = await predictLetter(results.multiHandLandmarks[0]);
    if (prediction) {
      displayText = `Predicted: ${prediction}`;
      if (prediction !== "unknown" && prediction !== lastTypedChar) {
        if (prediction.toLowerCase() === "backspace" && accumulatedText.length > 0) {
          accumulatedText = accumulatedText.slice(0, -1);
        } else if (prediction !== "unknown") { // Ensure "unknown" isn't appended
          accumulatedText += prediction;
        }
        lastTypedChar = prediction;
      }
    } else {
      displayText = `Collecting frames: ${sequence.length}/${SEQUENCE_LENGTH_LETTERS}`;
    }
  } else {
    // Auto-space logic when hand is not detected for a while
    const PAUSE_THRESHOLD_MS = 2000; // 2 seconds
    if (Date.now() - lastHandTime > PAUSE_THRESHOLD_MS && accumulatedText && !accumulatedText.endsWith(" ")) {
      accumulatedText += " ";
      lastHandTime = Date.now(); // Reset timer after adding space
    }
    // Clear sequences if hand is lost for robustness, similar to Python
    sequence = [];
    predictions = [];
    lastTypedChar = ""; // Reset last typed char too
  }
  updateUI(displayText);
}

// ===============================
// HOLISTIC (PHRASES) RESULTS HANDLER
// ===============================
async function onHolisticResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.scale(-1, 1); // Flip horizontally for selfie-view
  canvasCtx.drawImage(results.image, -canvasElement.width, 0, canvasElement.width, canvasElement.height);
  canvasCtx.restore();

  // Draw landmarks - USING WINDOW REFERENCES FOR SAFETY
  if (window.mp_drawing && window.mp_holistic) {
    if (results.faceLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.faceLandmarks, window.mp_holistic.FACEMESH_TESSELATION, 
                                     {color: '#808080', lineWidth: 1});
    }
    if (results.poseLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.poseLandmarks, window.mp_holistic.POSE_CONNECTIONS,
                                     {color: '#00FF00', lineWidth: 2});
      window.mp_drawing.drawLandmarks(canvasCtx, results.poseLandmarks, 
                                    {color: '#FF0000', lineWidth: 5});
    }
    if (results.leftHandLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.leftHandLandmarks, window.mp_holistic.HAND_CONNECTIONS,
                                     {color: '#00FFFF', lineWidth: 2});
      window.mp_drawing.drawLandmarks(canvasCtx, results.leftHandLandmarks, 
                                    {color: '#FF00FF', lineWidth: 5});
    }
    if (results.rightHandLandmarks) {
      window.mp_drawing.drawConnectors(canvasCtx, results.rightHandLandmarks, window.mp_holistic.HAND_CONNECTIONS,
                                     {color: '#FFFF00', lineWidth: 2});
      window.mp_drawing.drawLandmarks(canvasCtx, results.rightHandLandmarks, 
                                    {color: '#00FF00', lineWidth: 5});
    }
  }

  const keypoints = [];
  const hasPose = !!results.poseLandmarks;
  const hasFace = !!results.faceLandmarks;
  const hasLH = !!results.leftHandLandmarks;
  const hasRH = !!results.rightHandLandmarks;
  const anyDetected = hasPose || hasFace || hasLH || hasRH;

  if (hasPose) results.poseLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z, l.visibility));
  else keypoints.push(...Array(33 * 4).fill(0));

  if (hasFace) results.faceLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z));
  else keypoints.push(...Array(468 * 3).fill(0));

  if (hasLH) results.leftHandLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z));
  else keypoints.push(...Array(21 * 3).fill(0));

  if (hasRH) results.rightHandLandmarks.forEach(l => keypoints.push(l.x, l.y, l.z));
  else keypoints.push(...Array(21 * 3).fill(0));

  let prediction = null;
  if (anyDetected) {
    const normalizedKeypoints = normalizeHolisticKeypoints(keypoints);
    prediction = await predictPhrase(normalizedKeypoints); // This handles the 'none' state internally
    if (prediction && prediction !== "none") {
      if (prediction !== lastPhraseWord) {
        if (accumulatedText && !accumulatedText.endsWith(" ")) accumulatedText += " ";
        accumulatedText += prediction;
        lastPhraseWord = prediction;
      }
    } else if (prediction === "none") {
      // If the model predicts 'none', reset lastPhraseWord to allow re-detection
      lastPhraseWord = "";
    }
  } else {
    // If no holistic landmarks detected at all, ensure phrase mode state is 'none'
    prediction = await predictPhrase(null); // Pass null to trigger 'none' state
    lastPhraseWord = ""; // Clear last phrase word
  }
  updateUI(prediction || "None"); // Show 'None' if prediction is null/undefined
}

// ===============================
// INITIALIZE
// ===============================
(async () => {
  try {
    await loadModel(currentLanguage, mode);
    await initCamera();
    // Set initial active button
    if (mode === "letters") {
      document.getElementById("btn-letters").classList.add("active");
    } else {
      document.getElementById("btn-phrases").classList.add("active");
    }
    updateUI("Waiting for detection...");
  } catch (error) {
    console.error('Initialization error:', error);
    updateUI('Initialization failed. Check console for errors.');
  }
})();

// ===============================
// DEBUG EXPORTS (Press D to dump sequence, P to log probs)
// ===============================
window.addEventListener('keydown', async (e) => {
  if (e.key.toLowerCase() === 'd') {
    const seq = mode === 'letters' ? sequence : phraseSequence;
    const neededLength = mode === 'letters' ? SEQUENCE_LENGTH_LETTERS : SEQUENCE_LENGTH_PHRASES;
    if (seq.length === neededLength) {
      const blob = new Blob([JSON.stringify({ mode, language: currentLanguage, seq })], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `debug_sequence_${mode}_${currentLanguage}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      console.log('Exported sequence for debug');
    } else {
      console.warn(`Sequence not full yet. Current: ${seq.length}/${neededLength}`);
    }
  }
  if (e.key.toLowerCase() === 'p') {
    try {
      const seq = mode === 'letters' ? sequence : phraseSequence;
      const neededLength = mode === 'letters' ? SEQUENCE_LENGTH_LETTERS : SEQUENCE_LENGTH_PHRASES;
      if (seq.length !== neededLength) { console.warn('Sequence not full yet.'); return; }
      const input = tf.tensor([seq]);
      const pred = model.predict(input);
      const data = await pred.data();
      tf.dispose([input, pred]);
      console.log('Current probabilities:', Array.from(data));
      // Log class names with probabilities for phrases
      if (mode === 'phrases') {
        const sortedProbs = Array.from(data).map((prob, idx) => ({ action: labelMap[idx], prob }))
                               .sort((a, b) => b.prob - a.prob);
        console.log('Sorted Phrase Probabilities:');
        sortedProbs.forEach(item => console.log(`  ${item.action}: ${(item.prob * 100).toFixed(2)}%`));
      } else { // For letters
        const sortedProbs = Array.from(data).map((prob, idx) => ({ letter: labelMap[idx], prob }))
                               .sort((a, b) => b.prob - a.prob);
        console.log('Sorted Letter Probabilities:');
        sortedProbs.forEach(item => console.log(`  ${item.letter}: ${(item.prob * 100).toFixed(2)}%`));
      }

    } catch (err) {
      console.warn('Cannot compute probabilities:', err);
    }
  }
});
