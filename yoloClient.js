/**
 * YOLO Client
 * Calls the Python YOLO detection service at /detect
 * Falls back gracefully if service is unavailable
 */
require('dotenv').config()
const axios = require('axios')

const YOLO_URL = process.env.YOLO_SERVICE_URL || 'http://localhost:5001'

async function detect(imageBase64 = null) {
  try {
    const payload = imageBase64 ? { image: imageBase64 } : {}
    const res = await axios.post(`${YOLO_URL}/detect`, payload, { timeout: 1500 })
    return {
      humanDetected: res.data.humanDetected || false,
      smokeDetected: res.data.smokeDetected || false,
      debrisDetected: res.data.debrisDetected || false,
      detections: res.data.detections || [],
      yoloActive: true,
    }
  } catch {
    // YOLO service not running — return simulated detections
    return {
      humanDetected: Math.random() > 0.6,
      smokeDetected: Math.random() > 0.7,
      debrisDetected: Math.random() > 0.5,
      detections: [],
      yoloActive: false,
    }
  }
}

module.exports = { detect }
