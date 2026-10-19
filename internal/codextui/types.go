// Package codextui provides a display-only Codex workspace model. It does not
// start processes, read history, call an API, or answer approval requests.
// Adapters must translate their protocol into these presentation messages.
package codextui

type Kind string

const (
	Message Kind = "message"
	Command Kind = "command"
	Tool    Kind = "tool"
	Files   Kind = "files"
	Plan    Kind = "plan"
)

type Status string

const (
	Running     Status = "running"
	Completed   Status = "completed"
	Failed      Status = "failed"
	Declined    Status = "declined"
	Interrupted Status = "interrupted"
	Incomplete  Status = "incomplete"
)

// Capabilities describe available display data, never permission to act.
type Capabilities struct {
	ToolOutput bool
	Diffs      bool
	Plans      bool
}

type Config struct {
	Workspace     string
	Model         string
	Capabilities  Capabilities
	NoColor       bool
	ReducedMotion bool
}

// Scope is assigned by the adapter. Generation changes on every new run or
// connection; Sequence increases within a generation. IDs remain opaque.
type Scope struct {
	Generation uint64
	ThreadID   string
	TurnID     string
	Sequence   uint64
}

type BeginTurnMsg struct{ Scope }

type FileChange struct {
	Path string
	Kind string
}

// Item is a bounded display snapshot, not a wire-protocol type. Text is the
// current full text or command output. Diff must be supplied by the adapter;
// file names alone are never enough to infer a diff.
type Item struct {
	ID     string
	Kind   Kind
	Title  string
	Text   string
	Diff   string
	Status Status
	Files  []FileChange
}

// ItemMsg replaces an item's snapshot. A completed snapshot is authoritative
// over preceding deltas. Items are keyed by ID within this active turn.
// Non-regression applies to retained items only. The adapter must enforce
// lifecycle monotonicity for IDs evicted from this bounded display window.
type ItemMsg struct {
	Scope
	Item Item
}

// TextDeltaMsg appends to an existing non-terminal item's text. The adapter
// must emit an ItemMsg first; missing items are not silently invented.
type TextDeltaMsg struct {
	Scope
	ItemID string
	Delta  string
}

// EndTurnMsg accepts only the defined terminal Status constants.
type EndTurnMsg struct {
	Scope
	Status Status
}

// FrameMsg advances only cosmetic animation. No wall-clock access is needed by
// the reducer or View; the host owns scheduling and stops it when appropriate.
type FrameMsg struct{}
