/**
 * Real LLM Recommendation Engine
 * Uses Groq API (free, ultra-fast) with llama3-8b
 * Get free key at: https://console.groq.com
 */
require('dotenv').config()
const https = require('https')

const GROQ_API_KEY = process.env.GROQ_API_KEY || ''
const MODEL = 'llama-3.1-8b-instant'  // Fast, free, high rate limit

function buildPrompt(data) {
  const u1 = data.ultrasonic1 || data.distance || 200
  const u2 = data.ultrasonic2 || 200
  const u3 = data.ultrasonic3 || 200
  const humanNearby = data.humanNearby || (Math.min(u1, u2, u3) < 80)

  return `You are an AI assistant for the NDRF OmniRover disaster rescue system.
Analyze the following real-time sensor data and provide a concise tactical rescue recommendation in 2-3 sentences.
Be direct, military-style, actionable. No fluff.

SENSOR DATA:
- Temperature: ${data.temperature}°C ${data.temperature > 40 ? '(CRITICAL)' : data.temperature > 35 ? '(WARNING)' : '(NORMAL)'}
- Gas Level: ${data.gas} ppm ${data.gas > 300 ? '(TOXIC - evacuate)' : data.gas > 200 ? '(ELEVATED)' : '(SAFE)'}
- Ultrasonic Front: ${u1} cm ${u1 < 50 ? '(OBSTACLE/HUMAN CLOSE)' : u1 < 80 ? '(OBJECT NEARBY)' : '(CLEAR)'}
- Ultrasonic Left: ${u2} cm ${u2 < 80 ? '(OBJECT NEARBY)' : '(CLEAR)'}
- Ultrasonic Right: ${u3} cm ${u3 < 80 ? '(OBJECT NEARBY)' : '(CLEAR)'}
- Human Nearby (ultrasonic): ${humanNearby ? 'YES - within 80cm' : 'No'}
- Human Detected (camera AI): ${data.humanDetected ? 'YES - CONFIRMED SURVIVOR' : 'No'}
- Smoke/Fire: ${data.smokeDetected ? 'YES - FIRE RISK' : 'No'}
- Debris: ${data.debrisDetected ? 'YES' : 'No'}
- Overall Risk: ${data.risk}

Provide tactical rescue recommendation:`
}

async function getLLMRecommendation(sensorData) {
  // Compute risk for the prompt
  const risk = sensorData.gas > 300 || sensorData.temperature > 40 || sensorData.distance < 50
    ? 'DANGER'
    : sensorData.gas > 200 || sensorData.temperature > 35 || sensorData.distance < 100
    ? 'WARNING' : 'SAFE'
  const dataWithRisk = { ...sensorData, risk }

  if (!GROQ_API_KEY) return getFallbackRecommendation(dataWithRisk)

  return new Promise((resolve) => {
    const body = JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: buildPrompt(dataWithRisk) }],
      max_tokens: 150,
      temperature: 0.3,
    })

    const options = {
      hostname: 'api.groq.com',
      path: '/openai/v1/chat/completions',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Length': Buffer.byteLength(body),
      },
    }

    const req = https.request(options, (res) => {
      let data = ''
      res.on('data', chunk => data += chunk)
      res.on('end', () => {
        try {
          const json = JSON.parse(data)
          const text = json.choices?.[0]?.message?.content?.trim()
          resolve(text || getFallbackRecommendation(dataWithRisk))
        } catch {
          resolve(getFallbackRecommendation(sensorData))
        }
      })
    })

    req.on('error', () => resolve(getFallbackRecommendation(dataWithRisk)))
    req.setTimeout(3000, () => { req.destroy(); resolve(getFallbackRecommendation(dataWithRisk)) })
    req.write(body)
    req.end()
  })
}

function getFallbackRecommendation(data) {
  const { temperature, gas, distance, humanDetected, smokeDetected, risk } = data
  if (risk === 'DANGER') {
    if (gas > 300) return 'CRITICAL: Toxic gas levels exceed safe threshold. Deploy gas masks immediately. Initiate evacuation protocol. Do not send personnel without full HAZMAT gear.'
    if (temperature > 40) return 'CRITICAL: Thermal hazard detected. Risk of heat stroke for survivors. Deploy cooling units. Prioritize extraction within 10 minutes.'
    return 'CRITICAL: Multiple hazards detected. Halt rover advance. Request aerial support. Establish safe perimeter at 200m.'
  }
  if (humanDetected) return 'SURVIVOR DETECTED: Human life sign confirmed. Proceed with caution. Deploy rescue team to rover coordinates. Estimated extraction window: 15 minutes.'
  if (smokeDetected) return 'WARNING: Smoke detected indicating fire risk. Maintain safe distance. Deploy fire suppression team. Monitor temperature trends closely.'
  if (distance < 100) return 'WARNING: Obstacle detected ahead. Reduce rover speed. Navigate around obstruction. Continue systematic search pattern.'
  return 'ZONE CLEAR: All sensors nominal. Continue systematic search pattern. Maintain current heading. Report status every 5 minutes.'
}

module.exports = { getLLMRecommendation }

