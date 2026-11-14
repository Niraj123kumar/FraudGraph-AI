import { Transform } from 'node:stream'

export const MAX_UPLOAD_BYTES = 1_000_000_000
// Allow multipart boundaries and small form fields in addition to the CSV bytes.
export const MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 10_000_000

export function createLimitedBodyStream(source, maxBytes = MAX_BODY_BYTES) {
  let receivedBytes = 0
  let exceeded = false
  const body = source.pipe(new Transform({
    transform(chunk, _encoding, callback) {
      receivedBytes += chunk.length
      if (receivedBytes > maxBytes) {
        exceeded = true
        callback(new Error('Request body exceeds the configured limit'))
        return
      }
      callback(null, chunk)
    },
  }))

  return { body, exceeded: () => exceeded, receivedBytes: () => receivedBytes }
}
