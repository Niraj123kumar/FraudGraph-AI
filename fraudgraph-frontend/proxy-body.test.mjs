import assert from 'node:assert/strict'
import { Readable } from 'node:stream'
import test from 'node:test'

import { createLimitedBodyStream, MAX_BODY_BYTES, MAX_UPLOAD_BYTES } from './proxy-body.mjs'

async function collect(stream) {
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  return Buffer.concat(chunks)
}

test('proxy body cap includes multipart overhead beyond a 1 GB CSV', () => {
  assert.equal(MAX_UPLOAD_BYTES, 1_000_000_000)
  assert.ok(MAX_BODY_BYTES > MAX_UPLOAD_BYTES)
  assert.ok(MAX_BODY_BYTES <= MAX_UPLOAD_BYTES + 10_000_000)
})

test('limited stream passes content exactly at its configured cap', async () => {
  const limited = createLimitedBodyStream(Readable.from([Buffer.from('1234'), Buffer.from('5678')]), 8)
  assert.equal((await collect(limited.body)).toString(), '12345678')
  assert.equal(limited.exceeded(), false)
})

test('limited stream stops once it exceeds its configured cap', async () => {
  const limited = createLimitedBodyStream(Readable.from([Buffer.from('1234'), Buffer.from('56789')]), 8)
  await assert.rejects(() => collect(limited.body), /exceeds the configured limit/)
  assert.equal(limited.exceeded(), true)
})
