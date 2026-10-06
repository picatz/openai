// Package terminal contains input handling shared by the legacy terminal modes.
package terminal

import (
	"context"
	"io"
	"os"

	"github.com/muesli/cancelreader"
)

type input struct {
	reader   cancelreader.CancelReader
	ctx      context.Context
	stop     func() bool
	canceled chan struct{}
}

// NewInput interrupts an idle terminal read when ctx is canceled. Closing the
// reader releases its cancellation resources without closing the caller's file.
func NewInput(ctx context.Context, file *os.File) (io.ReadCloser, error) {
	reader, err := cancelreader.NewReader(file)
	if err != nil {
		return nil, err
	}
	r := &input{reader: reader, ctx: ctx, canceled: make(chan struct{})}
	r.stop = context.AfterFunc(ctx, func() { defer close(r.canceled); reader.Cancel() })
	return r, nil
}

func (r *input) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	n, err := r.reader.Read(p)
	if canceled := r.ctx.Err(); canceled != nil {
		return 0, canceled
	}
	// Raw terminals deliver Ctrl-C as a byte rather than a process signal. Keep
	// it distinguishable from Ctrl-D/EOF so the CLI can return exit code 130.
	for _, b := range p[:n] {
		if b == 3 {
			return 0, context.Canceled
		}
	}
	return n, err
}

func (r *input) Close() error {
	if !r.stop() {
		<-r.canceled
	}
	return r.reader.Close()
}
