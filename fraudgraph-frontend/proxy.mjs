import http from 'node:http'
import { createLimitedBodyStream } from './proxy-body.mjs'

const PORT = 8001
const HOST = '127.0.0.1'

const BACKEND_URL = process.env.FRAUDGRAPH_BACKEND_URL
const API_KEY = process.env.FRAUDGRAPH_API_KEY

let backendBase
try {
  backendBase = new URL(BACKEND_URL || '')
} catch {
  backendBase = null
}

if (!backendBase || !['http:', 'https:'].includes(backendBase.protocol) || backendBase.username || backendBase.password || !API_KEY) {
  console.error(
    'Missing FRAUDGRAPH_BACKEND_URL or FRAUDGRAPH_API_KEY.'
  )
  process.exit(1)
}

const server = http.createServer(async (req, res) => {
  if (!['GET', 'POST', 'PATCH', 'DELETE'].includes(req.method || '')) {
    res.writeHead(405)
    res.end('Method not allowed')
    return
  }

  let bodyExceeded = () => false
  try {
    if (!req.url?.startsWith('/') || req.url.startsWith('//')) {
      res.writeHead(400)
      res.end('Invalid request path')
      return
    }
    const backendUrl = new URL(req.url, backendBase)
    if (backendUrl.origin !== backendBase.origin) {
      res.writeHead(400)
      res.end('Invalid request origin')
      return
    }

    const limitedBody = req.method === 'GET' ? undefined : createLimitedBodyStream(req)
    if (limitedBody) bodyExceeded = limitedBody.exceeded
    const body = limitedBody?.body

    const headers = {
      'X-API-Key': API_KEY,
    }

    if (req.headers['content-type']) {
      headers['Content-Type'] = req.headers['content-type']
    }

    const response = await fetch(backendUrl, {
      method: req.method,
      headers,
      body,
      duplex: body ? 'half' : undefined,
      redirect: 'manual',
      signal: AbortSignal.timeout(1800000),
    })

    const responseBody = await response.arrayBuffer()

    res.writeHead(response.status, {
      'Content-Type':
        response.headers.get('content-type') || 'application/json',
      'Cache-Control': 'no-store',
    })

    res.end(Buffer.from(responseBody))
  } catch (error) {
    console.error('Proxy request failed:', error.message)

    if (bodyExceeded()) {
      res.writeHead(413)
      res.end('Request body too large')
      return
    }

    res.writeHead(502, {
      'Content-Type': 'application/json',
    })

    res.end(
      JSON.stringify({
        detail: 'Backend connection failed',
      })
    )
  }
})

server.listen(PORT, HOST, () => {
  console.log(`FraudGraph proxy listening on http://${HOST}:${PORT}`)
})
