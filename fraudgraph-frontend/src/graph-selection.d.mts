type SelectedTransfer = { source: string; target: string; evidence: { source_row: number } }
type DirectedEdge = { source: string; target: string }

export function isSelectedGraphEdge(edge: DirectedEdge, selectedTransfer: SelectedTransfer | null): boolean
export function isSelectedTransferAccount(accountId: string, selectedTransfer: SelectedTransfer | null): boolean
