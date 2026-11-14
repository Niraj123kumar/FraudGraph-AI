import assert from 'node:assert/strict'
import test from 'node:test'

import { isSelectedGraphEdge, isSelectedTransferAccount } from './src/graph-selection.mjs'

test('selecting a transfer highlights only its directed source-to-destination edge', () => {
  const selected = { source: 'ACCOUNT_SOURCE_01', target: 'ACCOUNT_DESTINATION_02', evidence: { source_row: 77 } }
  const edges = [
    { source: 'ACCOUNT_SOURCE_01', target: 'ACCOUNT_DESTINATION_02' },
    { source: 'ACCOUNT_DESTINATION_02', target: 'ACCOUNT_SOURCE_01' },
    { source: 'ACCOUNT_SOURCE_01', target: 'ACCOUNT_OTHER_03' },
  ]
  assert.deepEqual(edges.map((edge) => isSelectedGraphEdge(edge, selected)), [true, false, false])
  assert.equal(isSelectedTransferAccount('ACCOUNT_SOURCE_01', selected), true)
  assert.equal(isSelectedTransferAccount('ACCOUNT_DESTINATION_02', selected), true)
  assert.equal(isSelectedTransferAccount('ACCOUNT_OTHER_03', selected), false)
  assert.equal(isSelectedGraphEdge(edges[0], null), false)
})
