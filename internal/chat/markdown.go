package chat

import (
	"fmt"

	"github.com/charmbracelet/glamour"
)

func renderMarkdown(s string, width int) (string, error) {
	r, err := glamour.NewTermRenderer(
		glamour.WithStylePath("dark"),
		glamour.WithWordWrap(width),
		glamour.WithPreservedNewLines(),
	)
	if err != nil {
		return fmt.Errorf("failed to create markdown renderer: %w", err).Error(), nil
	}

	out, err := r.Render(s)
	if err != nil {
		return fmt.Errorf("failed to render markdown: %w", err).Error(), nil
	}

	return out, nil
}
