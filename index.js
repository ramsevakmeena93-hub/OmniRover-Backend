require('dotenv').config()
const express = require('express')
const http = require('http')
const WebSocket = require('ws')
const cors = require('cors')
const sensorManager = require('./sensorManager')
const { getLLMRecommendation, getVisionDescription } = require('./llm')

const app = express()
app.use(cors())
app.use(express.json({ limit: '10mb' }))

const server = http.createServer(app)

// ── WebSocket servers ──────────────────────────────────────────
const wss         = new WebSocket.Server({ noServer: true }) // frontend clients
const wssHardware = new WebSocket.Server({ noServer: true }) // ESP32 rover

// Route upgrades by path
server.on('upgrade', (req, socket, head) => {
  console.log('[UPGRADE] WebSocket path:', req.url)
  if (req.url === '/hardware') {
    wssHardware.handleUpgrade(req, socket, head, ws => wssHardware.emit('connection', ws, req))
  } else {
    wss.handleUpgrade(req, socket, head, ws => wss.emit('connection', ws, req))
  }
})

// Broadcast to all frontend clients
function broadcast(data) {
  const msg = JSON.stringify(data)
  wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(msg) })
}

// ── FRONTEND WebSocket ─────────────────────────────────────────
wss.on('connection', (ws) => {
  console.log('[WS] Frontend connected. Total:', wss.clients.size)
  // Send current state immediately
  ws.send(JSON.stringify(sensorManager.getLatest()))
  ws.on('close', () => console.log('[WS] Frontend disconnected'))
})

// ── ESP32 HARDWARE WebSocket ───────────────────────────────────
let lastLLMCall = 0
let lastRecommendation = 'Waiting for rover connection...'

wssHardware.on('connection', (ws) => {
  console.log('[HARDWARE] ✓ ESP32 rover connected!')

  ws.on('message', async (msg) => {
    try {
      const raw = JSON.parse(msg)
      console.log('[HARDWARE] Data:', JSON.stringify(raw))

      // Normalize field names — ESP32 sends "temp", website expects "temperature"
      const normalized = {
        temperature:      raw.temperature ?? raw.temp ?? null,
        humidity:         raw.humidity ?? null,
        gas:              raw.gas ?? null,
        distance:         raw.distance ?? raw.ultrasonic1 ?? null,
        ultrasonic1:      raw.ultrasonic1 ?? raw.distance ?? null,
        ultrasonic2:      raw.ultrasonic2 ?? null,
        ultrasonic3:      raw.ultrasonic3 ?? null,
        flameDetected:    raw.flame === 'DETECTED' || raw.flameDetected === true,
        humanNearby:      raw.humanNearby ?? ((raw.distance ?? 400) < 80),
        hardwareConnected: true,
      }

      // Pass normalized data to sensor manager
      sensorManager.onHardwareData(normalized)
      const data = sensorManager.getLatest()

      // LLM recommendation every 10 seconds
      const now = Date.now()
      if (now - lastLLMCall > 10000) {
        lastLLMCall = now
        getLLMRecommendation(data).then(rec => {
          lastRecommendation = rec
          sensorManager.setLatest({ ...sensorManager.getLatest(), recommendation: rec })
          broadcast(sensorManager.getLatest())
        }).catch(() => {})
      }

      const enriched = {
        ...data,
        recommendation: lastRecommendation,
        timestamp: new Date().toISOString()
      }
      sensorManager.setLatest(enriched)
      broadcast(enriched)

    } catch (e) {
      console.error('[HARDWARE] Parse error:', e.message)
    }
  })

  ws.on('close', () => {
    console.log('[HARDWARE] ESP32 disconnected')
    sensorManager.setLatest({
      ...sensorManager.getLatest(),
      hardwareConnected: false,
      recommendation: 'Rover disconnected — check WiFi connection'
    })
    broadcast(sensorManager.getLatest())
  })
})

// ── HTTP POST from ESP32 (simpler than WebSocket) ─────────────
app.post('/api/hardware', (req, res) => {
  try {
    const raw = req.body
    console.log('[HARDWARE] Data received:', JSON.stringify(raw))

    // Accept both "temp" and "temperature", "flame" string or bool
    const normalized = {
      temperature:   raw.temperature ?? raw.temp ?? null,
      humidity:      raw.humidity ?? null,
      gas:           raw.gas ?? null,
      distance:      raw.distance ?? raw.ultrasonic1 ?? null,
      ultrasonic1:   raw.ultrasonic1 ?? raw.distance ?? null,
      ultrasonic2:   raw.ultrasonic2 ?? null,
      ultrasonic3:   raw.ultrasonic3 ?? null,
      flameDetected: raw.flame === 'DETECTED' || raw.flameDetected === true,
      humanNearby:   raw.humanNearby ?? ((raw.distance ?? 400) < 80),
      hardwareConnected: true,
    }

    sensorManager.onHardwareData(normalized)

    // LLM every 10s
    const now = Date.now()
    if (now - lastLLMCall > 10000) {
      lastLLMCall = now
      getLLMRecommendation(normalized).then(rec => {
        lastRecommendation = rec
        sensorManager.setLatest({ ...sensorManager.getLatest(), recommendation: rec })
        broadcast(sensorManager.getLatest())
      }).catch(() => {})
    }

    const enriched = { ...sensorManager.getLatest(), recommendation: lastRecommendation, timestamp: new Date().toISOString() }
    sensorManager.setLatest(enriched)
    broadcast(enriched)

    res.json({ ok: true })
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

app.get('/api/health', (req, res) => res.json({
  status: 'ok',
  frontendClients: wss.clients.size,
  roverConnected: wssHardware.clients.size > 0
}))

// Command poll — ESP32 polls this to get latest command
let pendingCommand = { cmd: 'NONE', speed: 60, angle: 90 }
app.get('/api/command-poll', (req, res) => {
  res.json(pendingCommand)
  // Keep speed and angle for the consecutive polls
  pendingCommand = { cmd: 'NONE', speed: pendingCommand.speed, angle: pendingCommand.angle }
})

// Send command to ESP32 — send the full string format your ESP32 expects
const CMD_MAP = { 'FORWARD':'FORWARD', 'BACKWARD':'BACKWARD', 'LEFT':'LEFT', 'RIGHT':'RIGHT', 'STOP':'STOP', 'ROTATE_L':'LEFT', 'ROTATE_R':'RIGHT', 'NONE':'NONE' }
app.post('/api/command', (req, res) => {
  const { cmd, speed, angle } = req.body
  const esp32cmd = cmd ? (CMD_MAP[cmd] || 'STOP') : pendingCommand.cmd
  
  pendingCommand = { 
    cmd: esp32cmd, 
    speed: speed !== undefined ? speed : pendingCommand.speed,
    angle: angle !== undefined ? angle : pendingCommand.angle
  }
  
  console.log(`[CMD] Input:${cmd} → ESP32:${esp32cmd} | Speed: ${pendingCommand.speed} | Angle: ${pendingCommand.angle}`)
  // Send via WebSocket directly if ESP32 uses WebSocket
  let sent = 0
  wssHardware.clients.forEach(c => {
    if (c.readyState === WebSocket.OPEN) {
      c.send(JSON.stringify(pendingCommand))
      sent++
      console.log(`[CMD] Sent "${esp32cmd}" to ESP32 via WebSocket`)
    }
  })
  if (sent === 0) console.log('[CMD] No ESP32 WebSocket clients to send to (it might be HTTP polling instead)!')
  res.json({ ok: true, cmd: esp32cmd, angle: pendingCommand.angle })
})

// Vision LLM
let lastVisionCall = 0
app.post('/api/vision', async (req, res) => {
  const now = Date.now()
  if (now - lastVisionCall < 5000)
    return res.status(429).json({ error: 'Rate limited — wait 5 seconds' })
  lastVisionCall = now
  try {
    const { image } = req.body
    if (!image) return res.status(400).json({ error: 'No image provided' })
    const description = await getVisionDescription(image)
    res.json({ description, timestamp: new Date().toISOString() })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ── START ──────────────────────────────────────────────────────
sensorManager.start(() => {}) // no-op callback, data comes from ESP32

const PORT = process.env.PORT || 5000
server.listen(PORT, '0.0.0.0', () => {
  console.log(`[SERVER] Running on http://0.0.0.0:${PORT}`)
  console.log(`[SERVER] Local:   http://localhost:${PORT}`)
  console.log(`[SERVER] Network: http://10.222.71.222:${PORT}`)
  console.log('[SERVER] ESP32 connects to: ws://10.222.71.222:' + PORT + '/hardware')
})
