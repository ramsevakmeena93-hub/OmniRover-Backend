/**
 * Sensor Manager
 * - If USE_HARDWARE=true: reads from Arduino/RPi via SerialPort
 * - If USE_HARDWARE=false: generates realistic simulated data
 *
 * Arduino must send JSON lines like:
 * {"temperature":32.5,"gas":180,"distance":120,"lat":28.6139,"lng":77.2090}
 */

require('dotenv').config()
const USE_HARDWARE = process.env.USE_HARDWARE === 'true'

// Gwalior, MP — base coordinates (used when hardware not connected)
const BASE_LAT = 26.2183
const BASE_LNG = 78.1828

let latestData = {
  temperature: 30, gas: 150, distance: 200,
  ultrasonic1: 200, ultrasonic2: 200, ultrasonic3: 200, // front, left, right
  humanNearby: false,  // ultrasonic < 80cm = possible human
  lat: BASE_LAT, lng: BASE_LNG,
  humanDetected: false, smokeDetected: false, debrisDetected: false,
  timestamp: new Date().toISOString()
}

let intervalId = null
let serialPort = null
let parseBuffer = ''

// ── HARDWARE MODE ──────────────────────────────────────────────
function initSerial(onData) {
  const { SerialPort } = require('serialport')
  const { ReadlineParser } = require('@serialport/parser-readline')

  serialPort = new SerialPort({
    path: process.env.SERIAL_PORT || 'COM3',
    baudRate: parseInt(process.env.SERIAL_BAUD) || 9600,
  })

  const parser = serialPort.pipe(new ReadlineParser({ delimiter: '\n' }))

  serialPort.on('open', () => console.log(`[SERIAL] Connected to ${process.env.SERIAL_PORT}`))
  serialPort.on('error', (err) => console.error('[SERIAL] Error:', err.message))

  parser.on('data', (line) => {
    try {
      const parsed = JSON.parse(line.trim())
      onData(parsed)
    } catch (e) {
      // ignore malformed lines
    }
  })
}

// ── SIMULATION MODE ────────────────────────────────────────────
let _prev = null

function simulateSensor() {
  const prev = _prev || latestData
  const temp = Math.min(65, Math.max(20, parseFloat((prev.temperature + (Math.random() - 0.45) * 2).toFixed(1))))
  const gas = Math.min(300, Math.max(50, Math.round(prev.gas + (Math.random() - 0.5) * 10)))
  const dist = Math.min(300, Math.max(10, Math.round(prev.distance + (Math.random() - 0.5) * 15)))

  // Simulate 3 ultrasonic sensors (front, left, right)
  const u1 = Math.min(400, Math.max(5,  Math.round((prev.ultrasonic1 || 200) + (Math.random() - 0.5) * 20)))
  const u2 = Math.min(400, Math.max(5,  Math.round((prev.ultrasonic2 || 200) + (Math.random() - 0.5) * 20)))
  const u3 = Math.min(400, Math.max(5,  Math.round((prev.ultrasonic3 || 200) + (Math.random() - 0.5) * 20)))

  // Human nearby if any ultrasonic < 80cm (typical human detection range)
  const humanNearby = u1 < 80 || u2 < 80 || u3 < 80

  _prev = {
    temperature: temp, gas, distance: dist,
    ultrasonic1: u1, ultrasonic2: u2, ultrasonic3: u3,
    humanNearby,
    lat: BASE_LAT, lng: BASE_LNG,
    humanDetected: false,
    smokeDetected: false,
    debrisDetected: false,
  }
  return _prev
}

// ── PUBLIC API ─────────────────────────────────────────────────
function start(callback) {
  if (USE_HARDWARE) {
    console.log('[SENSOR] Hardware mode — reading from serial port')
    initSerial((data) => callback(data))
  } else {
    console.log('[SENSOR] Simulation mode — generating fake sensor data')
    intervalId = setInterval(() => {
      callback(simulateSensor())
    }, 2000)
  }
}

function getLatest() { return latestData }
function setLatest(data) { latestData = data }

module.exports = { start, getLatest, setLatest }
