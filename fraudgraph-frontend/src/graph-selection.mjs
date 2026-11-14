export function isSelectedGraphEdge(edge, selectedTransfer) {
  return Boolean(selectedTransfer && edge.source === selectedTransfer.source && edge.target === selectedTransfer.target)
}

export function isSelectedTransferAccount(accountId, selectedTransfer) {
  return Boolean(selectedTransfer && (accountId === selectedTransfer.source || accountId === selectedTransfer.target))
}
