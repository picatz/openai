// Package codexsession connects one fresh Codex app-server thread to a bounded
// presentation stream. It does not read history or implement approval handlers.
package codexsession

import "errors"

const (
	MaxPromptBytes = 64 << 10
	maxTextBytes   = 16 << 10
	maxItems       = 4096
	maxEvents      = 128
	maxIDBytes     = 512
)

var (
	ErrBusy               = errors.New("Codex turn already active")
	ErrNoTurn             = errors.New("no active Codex turn to interrupt")
	ErrProtocol           = errors.New("Codex app-server returned an invalid runtime payload")
	ErrLimit              = errors.New("Codex display stream exceeded its limit; result incomplete")
	ErrClosed             = errors.New("Codex workspace closed")
	ErrStartUncertain     = errors.New("Codex turn start was not confirmed; outcome unknown; reopen the workspace before sending again")
	ErrInterruptUncertain = errors.New("Codex interrupt was not confirmed; the turn may still be running")
)

// Options deliberately cannot override sandbox, approval, authentication, or
// experimental settings. Installed Codex configuration remains authoritative.
type Options struct{ Directory, Model string }

// Info reports the server's configured settings, not per-turn model telemetry.
// Only summaries are retained; Codex home/history paths are never opened.
type Info struct{ ThreadID, CWD, Model, ApprovalPolicy, ApprovalsReviewer, Sandbox string }
type FileChange struct{ Path, Kind string }
type Item struct {
	ID, Kind, Title, Text, Diff, Status string
	Files                               []FileChange
}

// Event is a bounded, owned snapshot/delta. Kind is begin, item, delta, end, or
// notice. Generation and Sequence are local ordering, not wire fields.
type Event struct {
	Kind, ThreadID, TurnID        string
	Generation, Sequence          uint64
	Item                          Item
	ItemID, Delta, Status, Notice string
}
