require('dotenv').config()
const express = require('express')
const http = require('http')
const WebSocket = require('ws')
const cors = require('cors')
const sensorManager = require('./sensorManager')
const yoloClient = require('./yoloClient')
const { getLLMRecommendation } = require('./llm')

const app = express()
app.use(cors())
app.use(express.json())

const server = http.createServer(app)
const wss = new WebSocket.Server({ server })

// Broadcast to all connected WebSocket clients
function broadcast(data) {
  const msg = JSON.stringify(data)
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) client.send(msg)
  })
}

// REST endpoint (fallback / initial load)
app.get('/api/sensors', async (req, res) => {
  const data = sensorManager.getLatest()
  res.json(data)
})

// Camera detection endpoint — receives base64 frame, returns YOLO results
app.post('/api/detect', async (req, res) => {
  try {
    const { image } = req.body
    const detections = await yoloClient.detect(image)
    res.json(detections)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// Health check
app.get('/api/health', (req, res) => res.json({ status: 'ok', clients: wss.clients.size }))

// WebSocket connection
wss.on('connection', (ws) => {
  console.log('[WS] Client connected. Total:', wss.clients.size)
  // Send latest data immediately on connect
  ws.send(JSON.stringify(sensorManager.getLatest()))
  ws.on('close', () => console.log('[WS] Client disconnected'))
})

// Throttle LLM calls — only every 10 seconds to avoid rate limits
let lastLLMCall = 0
let lastRecommendation = 'ZONE CLEAR: All sensors nominal. Continue systematic search pattern.'

// Start sensor reading loop — pushes data every 2s
sensorManager.start(async (rawSensorData) => {
  try {
    const detections = await yoloClient.detect()
    const merged = { ...rawSensorData, ...detections }

    // Call LLM only every 10 seconds
    const now = Date.now()
    if (now - lastLLMCall > 10000) {
      lastLLMCall = now
      getLLMRecommendation(merged).then(rec => {
        lastRecommendation = rec
      }).catch(() => {})
    }

    const enriched = { ...merged, recommendation: lastRecommendation, timestamp: new Date().toISOString() }
    sensorManager.setLatest(enriched)
    broadcast(enriched)
  } catch (err) {
    const enriched = { ...rawSensorData, recommendation: lastRecommendation, timestamp: new Date().toISOString() }
    sensorManager.setLatest(enriched)
    broadcast(enriched)
  }
})

const PORT = process.env.PORT || 5000
server.listen(PORT, () => console.log(`[SERVER] Running on http://localhost:${PORT}`))
