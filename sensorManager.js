/**
 * Sensor Manager — REAL HARDWARE ONLY
 * Waits for ESP32 data via WebSocket /hardware
 * No simulation, no dummy data
 * ESP32 sends JSON: {"temperature":32,"gas":180,"ultrasonic1":120,...}
 */

const BASE_LAT = 26.2183
const BASE_LNG = 78.1828

// Initial state — zeros until ESP32 connects
let latestData = {
  temperature: null,
  humidity:    null,
  gas:         null,
  distance:    null,
  ultrasonic1: null,
  ultrasonic2: null,
  ultrasonic3: null,
  flameDetected: false,
  humanNearby:   false,
  hardwareConnected: false,
  humanDetected:  false,
  smokeDetected:  false,
  debrisDetected: false,
  lat: BASE_LAT,
  lng: BASE_LNG,
  timestamp: new Date().toISOString()
}

let callback = null

// Called by index.js when ESP32 sends data
function onHardwareData(raw) {
  latestData = {
    ...latestData,
    ...raw,
    hardwareConnected: true,
    timestamp: new Date().toISOString()
  }
  if (callback) callback(latestData)
}

// start() — just stores the callback, no simulation loop
function start(cb) {
  callback = cb
  console.log('[SENSOR] Hardware-only mode — waiting for ESP32...')
}

function getLatest() { return latestData }
function setLatest(data) { latestData = data }

module.exports = { start, getLatest, setLatest, onHardwareData }
