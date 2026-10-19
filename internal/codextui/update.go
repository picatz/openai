package codextui

import tea "charm.land/bubbletea/v2"

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = max(0, msg.Width), max(0, msg.Height)
		m.resize()
	case BeginTurnMsg:
		if msg.Generation <= m.scope.Generation || !validID(msg.ThreadID) || !validID(msg.TurnID) {
			return m, nil
		}
		m.scope, m.active, m.status = msg.Scope, true, Running
		m.items, m.selected, m.dropped, m.frame = nil, 0, 0, 0
		m.detail.GotoTop()
		m.refresh(false)
	case ItemMsg:
		if !m.accepts(msg.Scope) || !validID(msg.Item.ID) {
			return m, nil
		}
		item := cloneItem(msg.Item)
		if item.ID == "" {
			return m, nil
		}
		i := m.index(item.ID)
		if i >= 0 && terminal(m.items[i].Status) && !terminal(item.Status) {
			return m, nil
		}
		m.scope.Sequence = msg.Sequence
		bottom := m.detail.AtBottom()
		if i < 0 {
			m.items = append(m.items, item)
			if len(m.items) > maxItems {
				m.items = append([]Item(nil), m.items[1:]...)
				m.selected = max(0, m.selected-1)
				m.dropped++
			}
		} else {
			m.items[i] = item
		}
		m.refresh(bottom)
	case TextDeltaMsg:
		if !m.accepts(msg.Scope) {
			return m, nil
		}
		i := m.index(msg.ItemID)
		if i < 0 || terminal(m.items[i].Status) {
			return m, nil
		}
		m.scope.Sequence = msg.Sequence
		bottom := m.detail.AtBottom()
		// The cap is applied before concatenation so a single huge delta cannot
		// create an unbounded retained buffer.
		m.items[i].Text = limitText(m.items[i].Text+limitText(msg.Delta, maxTextBytes), maxTextBytes)
		m.refresh(bottom)
	case EndTurnMsg:
		if !m.accepts(msg.Scope) {
			return m, nil
		}
		switch msg.Status {
		case Completed, Failed, Declined, Interrupted, Incomplete:
		default:
			return m, nil
		}
		m.scope.Sequence, m.active, m.status = msg.Sequence, false, msg.Status
		for i := range m.items {
			if !terminal(m.items[i].Status) {
				m.items[i].Status = Incomplete
				if msg.Status == Interrupted {
					m.items[i].Status = Interrupted
				}
			}
		}
		m.refresh(false)
	case FrameMsg:
		if m.active && !m.cfg.ReducedMotion && !m.cfg.NoColor {
			m.frame = (m.frame + 1) % 4
		}
	case tea.KeyPressMsg:
		switch msg.String() {
		case "ctrl+q":
			return m, tea.Quit
		case "?", "f1":
			m.help = !m.help
			return m, nil
		case "esc":
			m.help = false
			m.focusDetail = false
			return m, nil
		}
		if m.help {
			return m, nil
		}
		switch msg.String() {
		case "tab", "shift+tab":
			m.focusDetail = !m.focusDetail
		case "up", "k", "down", "j", "home", "end":
			if m.focusDetail {
				m.detail, _ = m.detail.Update(msg)
				return m, nil
			}
			switch msg.String() {
			case "up", "k":
				m.selected = max(0, m.selected-1)
			case "down", "j":
				m.selected = min(max(0, len(m.items)-1), m.selected+1)
			case "home":
				m.selected = 0
			case "end":
				m.selected = max(0, len(m.items)-1)
			}
			m.detail.GotoTop()
			m.refresh(false)
		case "pgup", "pgdown":
			m.detail, _ = m.detail.Update(msg)
		case "ctrl+home":
			m.detail.GotoTop()
		case "ctrl+end":
			m.detail.GotoBottom()
		}
	case tea.MouseWheelMsg:
		if !m.help {
			m.detail, _ = m.detail.Update(msg)
		}
	}
	return m, nil
}
