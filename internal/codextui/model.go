package codextui

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

const (
	maxItems     = 128
	maxTextBytes = 16 << 10
	maxFiles     = 64
)

type Model struct {
	cfg                       Config
	scope                     Scope
	items                     []Item
	selected                  int
	width, height             int
	detail                    viewport.Model
	focusDetail, help, active bool
	status                    Status
	frame, dropped            int
}

func New(cfg Config) Model {
	cfg.Workspace = limitText(cfg.Workspace, 512)
	cfg.Model = limitText(cfg.Model, 256)
	m := Model{cfg: cfg, width: 80, height: 24, detail: viewport.New()}
	m.resize()
	return m
}

func (m Model) Init() tea.Cmd { return nil }

func safeText(s string) string {
	return strings.Map(func(r rune) rune {
		if (unicode.IsControl(r) && r != '\n' && r != '\t') || unicode.Is(unicode.Bidi_Control, r) {
			return -1
		}
		return r
	}, ansi.Strip(strings.ToValidUTF8(s, "�")))
}

func singleLine(s string) string { return strings.Join(strings.Fields(safeText(s)), " ") }

func limitText(s string, n int) string {
	// Bound the retained raw data before parsing terminal sequences.
	truncated := len(s) > n
	if truncated {
		s = s[:n]
	}
	s = safeText(s)
	// Replacing malformed UTF-8 can expand the normalized string even when the
	// raw bytes fit. Bound that result too, including the truncation label.
	truncated = truncated || len(s) > n
	if truncated {
		marker := "\n[display truncated]"
		if n < len(marker) {
			return marker[:max(0, n)]
		}
		if limit := n - len(marker); len(s) > limit {
			s = s[:limit]
			for !utf8.ValidString(s) && len(s) > 0 {
				s = s[:len(s)-1]
			}
		}
		s += marker
	}
	return s
}

func cloneItem(item Item) Item {
	// Keep IDs opaque; rejecting oversized IDs avoids lossy-key collisions.
	item.Kind = Kind(singleLine(limitText(string(item.Kind), 80)))
	item.Title = singleLine(limitText(item.Title, 512))
	item.Text = limitText(item.Text, maxTextBytes)
	item.Diff = limitText(item.Diff, maxTextBytes)
	item.Status = Status(singleLine(limitText(string(item.Status), 80)))
	n := min(len(item.Files), maxFiles)
	files := make([]FileChange, n)
	for i, file := range item.Files[:n] {
		files[i] = FileChange{Path: singleLine(limitText(file.Path, 512)), Kind: singleLine(limitText(file.Kind, 80))}
	}
	item.Files = files
	return item
}

func validID(id string) bool { return id != "" && len(id) <= 512 }

func terminal(status Status) bool { return status != "" && status != Running }

func (m Model) accepts(scope Scope) bool {
	return m.active && scope.Generation == m.scope.Generation && scope.ThreadID == m.scope.ThreadID &&
		scope.TurnID == m.scope.TurnID && scope.Sequence > m.scope.Sequence
}

func (m Model) index(id string) int {
	for i, item := range m.items {
		if item.ID == id {
			return i
		}
	}
	return -1
}

func (m *Model) resize() {
	w := max(1, m.width-4)
	if m.width >= 100 {
		w = max(1, m.width-m.width/3-7)
	}
	m.detail.SetWidth(w)
	m.detail.SetHeight(max(1, m.height-7))
	m.detail.SoftWrap = true
	m.refresh(false)
}

func (m *Model) refresh(bottom bool) {
	m.detail.SetContent(m.renderDetail())
	if bottom {
		m.detail.GotoBottom()
	}
}
