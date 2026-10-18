package codexworkspace

import (
	"context"
	"errors"
	"fmt"

	tea "charm.land/bubbletea/v2"
)

// Run owns Backend.Close, including pre-canceled contexts and startup failure.
// It overrides WithContext: normal Quit waits for input/terminal cleanup, unlike
// Bubble Tea v2.0.10's force-kill path. Closing locally is not remote completion.
func Run(cfg Config, options ...tea.ProgramOption) (err error) {
	if cfg.Backend == nil {
		return fmt.Errorf("codex workspace: missing backend")
	}
	defer func() { err = errors.Join(err, cfg.Backend.Close()) }()
	if cfg.Context == nil {
		cfg.Context = context.Background()
	}
	ctx, cancel := context.WithCancel(cfg.Context)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return err
	}
	cfg.Context = ctx
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
	model, runErr := program.Run()
	close(done)
	<-stopped
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if runErr != nil {
		return runErr
	}
	if m, ok := model.(Model); ok {
		return m.transportErr
	}
	return nil
}
