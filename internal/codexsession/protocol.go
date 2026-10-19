package codexsession

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"unicode/utf8"

	"github.com/picatz/openai/codex/appserver"
)

// Only this documented stable subset is decoded. Unknown notifications and
// item content are not rendered as raw JSON (which may contain credentials).
// No method in this package can answer an incoming server request.
type wireTurn struct {
	ID     string            `json:"id"`
	Status string            `json:"status"`
	Items  []json.RawMessage `json:"items"`
}
type wireFile struct {
	Path string `json:"path"`
	Kind struct {
		Type     string  `json:"type"`
		MovePath *string `json:"move_path"`
	} `json:"kind"`
	Diff string `json:"diff"`
}
type wireContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}
type wireItem struct {
	ID, Type, Text, Command, Status, Server, Tool string
	AggregatedOutput                              *string
	Changes                                       []wireFile
	Content                                       []wireContent
}

// Discriminate before decoding variant-specific fields. In particular reasoning
// content is []string, not userMessage content; future variants may use any
// shape. Unknown fields never inherit another variant's decoder.
func decodeItem(raw json.RawMessage) (wireItem, error) {
	var head struct {
		ID   string `json:"id"`
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &head) != nil || !validID(head.ID) || head.Type == "" || len(head.Type) > 128 {
		return wireItem{}, ErrProtocol
	}
	w := wireItem{ID: head.ID, Type: head.Type}
	switch w.Type {
	case "agentMessage":
		var p struct {
			Text *string `json:"text"`
		}
		if json.Unmarshal(raw, &p) != nil || p.Text == nil {
			return wireItem{}, ErrProtocol
		}
		w.Text = *p.Text
	case "userMessage":
		var p struct {
			Content []wireContent `json:"content"`
		}
		if json.Unmarshal(raw, &p) != nil || p.Content == nil {
			return wireItem{}, ErrProtocol
		}
		w.Content = p.Content
	case "commandExecution":
		var p struct {
			Command string  `json:"command"`
			Status  string  `json:"status"`
			Output  *string `json:"aggregatedOutput"`
		}
		if json.Unmarshal(raw, &p) != nil {
			return wireItem{}, ErrProtocol
		}
		w.Command = p.Command
		w.Status = p.Status
		w.AggregatedOutput = p.Output
	case "fileChange":
		var p struct {
			Changes []wireFile `json:"changes"`
			Status  string     `json:"status"`
		}
		if json.Unmarshal(raw, &p) != nil || p.Changes == nil {
			return wireItem{}, ErrProtocol
		}
		w.Changes = p.Changes
		w.Status = p.Status
	case "mcpToolCall":
		var p struct {
			Server string `json:"server"`
			Tool   string `json:"tool"`
			Status string `json:"status"`
		}
		if json.Unmarshal(raw, &p) != nil {
			return wireItem{}, ErrProtocol
		}
		w.Server = p.Server
		w.Tool = p.Tool
		w.Status = p.Status
	}
	return w, nil
}

func validID(s string) bool { return s != "" && len(s) <= maxIDBytes && utf8.ValidString(s) }
func validTurnStatus(s string) bool {
	return s == "inProgress" || s == "completed" || s == "interrupted" || s == "failed"
}
func bounded(s string, n int) string {
	if len(s) <= n {
		return s
	}
	const marker = "\n[display truncated]"
	i := max(0, n-len(marker))
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	return s[:i] + marker
}
func displayID(id string) string {
	sum := sha256.Sum256([]byte(id))
	return "item:" + hex.EncodeToString(sum[:])
}

func parseInfo(raw json.RawMessage) (Info, error) {
	var r struct {
		Thread struct {
			ID string `json:"id"`
		} `json:"thread"`
		CWD               string          `json:"cwd"`
		Model             string          `json:"model"`
		ApprovalPolicy    json.RawMessage `json:"approvalPolicy"`
		ApprovalsReviewer string          `json:"approvalsReviewer"`
		Sandbox           struct {
			Type string `json:"type"`
		} `json:"sandbox"`
	}
	if json.Unmarshal(raw, &r) != nil || !validID(r.Thread.ID) || r.CWD == "" || r.Model == "" || r.ApprovalsReviewer == "" || r.Sandbox.Type == "" || len(r.ApprovalPolicy) == 0 {
		return Info{}, ErrProtocol
	}
	policy := "configured"
	var value string
	if json.Unmarshal(r.ApprovalPolicy, &value) == nil {
		policy = bounded(value, 256)
	}
	return Info{ThreadID: r.Thread.ID, CWD: bounded(r.CWD, 512), Model: bounded(r.Model, 256), ApprovalPolicy: policy, ApprovalsReviewer: bounded(r.ApprovalsReviewer, 256), Sandbox: bounded(r.Sandbox.Type, 256)}, nil
}

func (s *Session) reduceLocked(e appserver.Event) error {
	if e.Rejected {
		return s.emitLocked(Event{Kind: "notice", Notice: "Unsupported server request rejected: " + bounded(e.Method, 160) + ". Use the native Codex client for approvals or input."})
	}
	switch e.Method {
	case "turn/started", "turn/completed":
		var p struct {
			ThreadID string   `json:"threadId"`
			Turn     wireTurn `json:"turn"`
		}
		if json.Unmarshal(e.Params, &p) != nil || !validID(p.ThreadID) || !validID(p.Turn.ID) || !validTurnStatus(p.Turn.Status) || p.Turn.Items == nil {
			return ErrProtocol
		}
		if !s.active || p.ThreadID != s.info.ThreadID || p.Turn.ID != s.turnID {
			return nil
		}
		if e.Method == "turn/started" {
			if p.Turn.Status != "inProgress" {
				return ErrProtocol
			}
			return nil
		}
		if p.Turn.Status == "inProgress" {
			return ErrProtocol
		}
		// Completion items may be only a last-message summary. Merge supported
		// supplied final items; never replace the existing timeline with it.
		for _, raw := range p.Turn.Items {
			if err := s.itemLocked(raw, true); err != nil {
				return err
			}
		}
		if err := s.emitLocked(Event{Kind: "end", Status: p.Turn.Status}); err != nil {
			return err
		}
		s.active = false
		s.items = nil
		return nil
	case "item/started", "item/completed":
		var p struct {
			ThreadID string          `json:"threadId"`
			TurnID   string          `json:"turnId"`
			Item     json.RawMessage `json:"item"`
		}
		if json.Unmarshal(e.Params, &p) != nil || !validID(p.ThreadID) || !validID(p.TurnID) || len(p.Item) == 0 {
			return ErrProtocol
		}
		if !s.matches(p.ThreadID, p.TurnID) {
			return nil
		}
		return s.itemLocked(p.Item, e.Method == "item/completed")
	case "item/agentMessage/delta", "item/commandExecution/outputDelta":
		var p struct {
			ThreadID string  `json:"threadId"`
			TurnID   string  `json:"turnId"`
			ItemID   string  `json:"itemId"`
			Delta    *string `json:"delta"`
		}
		if json.Unmarshal(e.Params, &p) != nil || !validID(p.ThreadID) || !validID(p.TurnID) || !validID(p.ItemID) || p.Delta == nil {
			return ErrProtocol
		}
		if !s.matches(p.ThreadID, p.TurnID) {
			return nil
		}
		state, ok := s.items[p.ItemID]
		if !ok {
			return ErrProtocol
		} // Never invent a missing lifecycle item.
		if state.terminal {
			return nil
		}
		if (e.Method == "item/agentMessage/delta" && state.kind != "agentMessage") || (e.Method == "item/commandExecution/outputDelta" && state.kind != "commandExecution") {
			return ErrProtocol
		}
		if state.truncated {
			return nil
		}
		delta := *p.Delta
		remain := maxTextBytes - state.bytes
		if len(delta) > remain {
			// Keep one bounded notice in the appended text. No accumulated text
			// is retained here, only a capped byte counter per lifecycle item.
			delta = bounded(delta, max(32, remain))
			state.truncated = true
		}
		state.bytes = min(maxTextBytes, state.bytes+len(delta))
		s.items[p.ItemID] = state
		if delta == "" {
			return nil
		}
		return s.emitLocked(Event{Kind: "delta", ItemID: displayID(p.ItemID), Delta: delta})
	case "item/fileChange/patchUpdated":
		var p struct {
			ThreadID string     `json:"threadId"`
			TurnID   string     `json:"turnId"`
			ItemID   string     `json:"itemId"`
			Changes  []wireFile `json:"changes"`
		}
		if json.Unmarshal(e.Params, &p) != nil || !validID(p.ThreadID) || !validID(p.TurnID) || !validID(p.ItemID) || p.Changes == nil {
			return ErrProtocol
		}
		if !s.matches(p.ThreadID, p.TurnID) {
			return nil
		}
		state, ok := s.items[p.ItemID]
		if !ok || state.kind != "fileChange" {
			return ErrProtocol
		}
		if state.terminal {
			return nil
		}
		return s.projectItemLocked(wireItem{ID: p.ItemID, Type: "fileChange", Status: "inProgress", Changes: p.Changes}, false)
	case "turn/diff/updated":
		var p struct {
			ThreadID string  `json:"threadId"`
			TurnID   string  `json:"turnId"`
			Diff     *string `json:"diff"`
		}
		if json.Unmarshal(e.Params, &p) != nil || !validID(p.ThreadID) || !validID(p.TurnID) || p.Diff == nil {
			return ErrProtocol
		}
		if !s.matches(p.ThreadID, p.TurnID) {
			return nil
		}
		return s.emitLocked(Event{Kind: "item", Item: Item{ID: "meta:diff", Kind: "files", Title: "Turn diff snapshot", Diff: bounded(*p.Diff, maxTextBytes), Status: "completed"}})
	case "turn/plan/updated":
		var p struct {
			ThreadID    string  `json:"threadId"`
			TurnID      string  `json:"turnId"`
			Explanation *string `json:"explanation"`
			Plan        []struct {
				Step   string `json:"step"`
				Status string `json:"status"`
			} `json:"plan"`
		}
		if json.Unmarshal(e.Params, &p) != nil || !validID(p.ThreadID) || !validID(p.TurnID) || p.Plan == nil {
			return ErrProtocol
		}
		if !s.matches(p.ThreadID, p.TurnID) {
			return nil
		}
		text := ""
		if p.Explanation != nil {
			text = bounded(*p.Explanation, maxTextBytes) + "\n"
		}
		for i, step := range p.Plan {
			if step.Status != "pending" && step.Status != "inProgress" && step.Status != "completed" {
				return ErrProtocol
			}
			if i >= 64 || len(text) >= maxTextBytes {
				text = bounded(text+"\n[additional plan steps omitted]", maxTextBytes)
				break
			}
			text = bounded(text+step.Status+"  "+bounded(step.Step, maxTextBytes)+"\n", maxTextBytes)
		}
		return s.emitLocked(Event{Kind: "item", Item: Item{ID: "meta:plan", Kind: "plan", Title: "Turn plan snapshot", Text: text, Status: "completed"}})
	case "error":
		// The stream may report retryable service errors. Never render raw
		// error data or infer completion from this notification.
		return s.emitLocked(Event{Kind: "notice", Notice: "Codex reported a service error. Waiting for confirmed turn status."})
	default:
		return nil
	}
}
func (s *Session) matches(thread, turn string) bool {
	return s.active && thread == s.info.ThreadID && turn == s.turnID
}

func (s *Session) itemLocked(raw json.RawMessage, complete bool) error {
	w, err := decodeItem(raw)
	if err != nil {
		return err
	}
	return s.projectItemLocked(w, complete)
}
func (s *Session) projectItemLocked(w wireItem, complete bool) error {
	state, exists := s.items[w.ID]
	if exists && state.terminal {
		return nil
	}
	if !exists && len(s.items) >= maxItems {
		return ErrLimit
	}
	if exists && state.kind != w.Type {
		return ErrProtocol
	}
	item := Item{ID: displayID(w.ID), Status: "running"}
	if complete {
		item.Status = "completed"
	}
	switch w.Type {
	case "agentMessage":
		item.Kind = "message"
		item.Title = "Codex"
		item.Text = bounded(w.Text, maxTextBytes)
	case "userMessage":
		item.Kind = "message"
		item.Title = "You"
		for _, part := range w.Content {
			if part.Type == "text" {
				item.Text = bounded(item.Text+bounded(part.Text, maxTextBytes), maxTextBytes)
			}
			if len(item.Text) >= maxTextBytes {
				break
			}
		}
	case "commandExecution":
		item.Kind = "command"
		item.Title = bounded(w.Command, 512)
		if w.AggregatedOutput != nil {
			item.Text = bounded(*w.AggregatedOutput, maxTextBytes)
		}
	case "fileChange":
		item.Kind = "files"
		item.Title = "Changed files"
		if len(w.Changes) > 64 {
			item.Title = "Changed files [showing at most 64 paths]"
		}
		for i, change := range w.Changes {
			if i >= 64 {
				break
			}
			path := bounded(change.Path, 512)
			if change.Kind.MovePath != nil {
				path = bounded(path+" → "+bounded(*change.Kind.MovePath, 512), 512)
			}
			item.Files = append(item.Files, FileChange{Path: path, Kind: bounded(change.Kind.Type, 80)})
			item.Diff = bounded(item.Diff+bounded(change.Diff, maxTextBytes), maxTextBytes)
		}
	case "mcpToolCall":
		item.Kind = "tool"
		item.Title = bounded(w.Server+" / "+w.Tool, 512)
		item.Text = "Tool payload is not displayed by this client."
	default:
		item.Kind = "tool"
		item.Title = "Unsupported item: " + bounded(w.Type, 160)
		item.Text = "Item details are unavailable in this client."
	}
	if w.Type == "commandExecution" || w.Type == "fileChange" || w.Type == "mcpToolCall" {
		switch w.Status {
		case "inProgress":
			if complete {
				return ErrProtocol
			}
			item.Status = "running"
		case "completed", "failed", "declined":
			item.Status = w.Status
		default:
			return ErrProtocol
		}
	}
	// A terminal snapshot is authoritative. Later running snapshots/deltas are
	// ignored for every item, including ones the smaller display window evicts.
	s.items[w.ID] = itemState{kind: w.Type, terminal: complete || item.Status != "running", bytes: len(item.Text), truncated: len(item.Text) >= maxTextBytes}
	return s.emitLocked(Event{Kind: "item", Item: item})
}
