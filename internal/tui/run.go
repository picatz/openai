package tui

import (
	"context"

	tea "charm.land/bubbletea/v2"
)

// Run owns the conversation context and gracefully restores the terminal when
// that context is canceled. Config.Context governs cancellation; any
// WithContext program options are overridden.
func Run(cfg Config, options ...tea.ProgramOption) error {
	if cfg.Context == nil {
		cfg.Context = context.Background()
	}
	ctx, cancel := context.WithCancel(cfg.Context)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return err
	}
	cfg.Context = ctx
	// WithContext would force-kill Bubble Tea on cancellation. In v2.0.10 that
	// closes the input reader without waiting for it, racing terminal cleanup.
	// Keep backend cancellation immediate but ask the event loop to quit normally.
	options = append(options, tea.WithContext(context.Background()))
	program := tea.NewProgram(New(cfg), options...)
	done, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		select {
		case <-ctx.Done():
			program.Quit()
		case <-done:
		}
	}()
	_, err := program.Run()
	close(done)
	<-stopped
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}
