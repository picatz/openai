package codexsession

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"
)

// processCloseGrace gives an app-server that observes stdin EOF a brief chance
// to exit before Close kills it. Only the direct child is owned; this is not
// process-tree supervision.
const processCloseGrace = 250 * time.Millisecond

// processPipe owns the direct child and its stdin/stdout. The endpoints are real
// files, so closing them interrupts concurrent reads and writes without waiting
// for a user-space copying goroutine. Call Close even after Wait reports exit.
//
// Process exit and stdout EOF are distinct. In particular, Wait does not close
// stdout, because a child may have exited with a final protocol frame buffered
// in the pipe. Conversely, stdout EOF alone does not mean the child has exited.
type processPipe struct {
	cmd     *exec.Cmd
	stdin   *os.File
	stdout  *os.File
	done    chan struct{}
	waitErr error // Written once, before done closes; read only after done.

	closeOnce sync.Once
	closed    chan struct{} // Closed after cleanup, including joining the child.
	closeErr  error         // Published by closeOnce to all Close callers.
}

var _ io.ReadWriteCloser = (*processPipe)(nil)

// startProcess launches an already-installed Codex CLI using its existing
// configuration and permissions. It neither installs Codex nor supplies config,
// authentication, sandbox, approval, or other policy overrides.
//
// ctx is explicitly the process lifetime context. A session must pass its own
// lifetime context, not a short-lived initialization or individual request
// context. Canceling ctx closes the transport and terminates the direct child.
func startProcess(ctx context.Context, directory string) (*processPipe, error) {
	if err := processContextError(ctx); err != nil {
		return nil, err
	}
	path, err := exec.LookPath("codex")
	if err != nil {
		return nil, fmt.Errorf("find Codex CLI: install Codex and make its executable available on PATH: %w", err)
	}
	cmd := exec.Command(path, "app-server")
	cmd.Dir = directory
	return startProcessCommand(ctx, cmd)
}

// startProcessCommand is the injection seam for fake-child tests. cmd must be
// unstarted, created with exec.Command (not exec.CommandContext), and must not
// already have standard streams assigned. After success only processPipe may
// call cmd.Wait or manage its process and standard streams.
func startProcessCommand(ctx context.Context, cmd *exec.Cmd) (*processPipe, error) {
	if err := processContextError(ctx); err != nil {
		return nil, err
	}
	if cmd == nil || cmd.Process != nil || cmd.Stdin != nil || cmd.Stdout != nil || cmd.Stderr != nil || cmd.Cancel != nil {
		return nil, errors.New("start Codex app-server: expected an unstarted command with unassigned standard streams")
	}

	// Use explicit pipes rather than StdoutPipe: Cmd.Wait must not close our
	// stdout reader before the transport drains the child's buffered output.
	childIn, parentIn, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("prepare Codex app-server stdin: %w", err)
	}
	parentOut, childOut, err := os.Pipe()
	if err != nil {
		_ = childIn.Close()
		_ = parentIn.Close()
		return nil, fmt.Errorf("prepare Codex app-server stdout: %w", err)
	}
	// A direct file retains zero stderr bytes and creates no exec.Cmd copy
	// goroutine that could hang if a descendant inherits stderr.
	stderr, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		_ = childIn.Close()
		_ = parentIn.Close()
		_ = parentOut.Close()
		_ = childOut.Close()
		return nil, fmt.Errorf("prepare Codex app-server stderr discard: %w", err)
	}
	defer childIn.Close()
	defer childOut.Close()
	defer stderr.Close()

	cmd.Stdin, cmd.Stdout, cmd.Stderr = childIn, childOut, stderr
	if err := cmd.Start(); err != nil {
		_ = parentIn.Close()
		_ = parentOut.Close()
		return nil, fmt.Errorf("start Codex app-server (check the executable and working directory): %w", err)
	}

	p := &processPipe{
		cmd: cmd, stdin: parentIn, stdout: parentOut,
		done: make(chan struct{}), closed: make(chan struct{}),
	}
	go func() {
		p.waitErr = cmd.Wait()
		close(p.done)
	}()
	if ctx.Done() != nil {
		go func() {
			select {
			case <-ctx.Done():
				_ = p.Close()
			case <-p.closed:
			}
		}()
	}
	return p, nil
}

func processContextError(ctx context.Context) error {
	if ctx == nil {
		return errors.New("start Codex app-server: a process lifetime context is required")
	}
	return ctx.Err()
}

func (p *processPipe) Read(b []byte) (int, error)  { return p.stdout.Read(b) }
func (p *processPipe) Write(b []byte) (int, error) { return p.stdin.Write(b) }

// Done closes only after the direct child has been waited for and reaped. It is
// safe for any number of observers. It does not indicate stdout EOF or Close.
func (p *processPipe) Done() <-chan struct{} { return p.done }

// Wait returns the original direct-child exit result, including a nonzero exit
// or a signal caused by Close. All callers receive the same result. Wait never
// waits for stdout EOF, descendant processes, or the context observer to exit.
func (p *processPipe) Wait() error {
	<-p.done
	return p.waitErr
}

// Close closes both transport endpoints first, unblocking concurrent Read and
// Write calls, then gives the child a short grace period to exit. If needed it
// kills and waits for the direct child. Concurrent and repeated calls join the
// same cleanup. Only cleanup errors are returned; use Wait for the exit result.
//
// The grace interval is bounded. Reaping after a successful kill is mandatory,
// rather than abandoning the child in exchange for a hard wall-clock deadline.
func (p *processPipe) Close() error {
	p.closeOnce.Do(func() {
		defer close(p.closed)
		p.closeErr = errors.Join(closeProcessFile(p.stdin), closeProcessFile(p.stdout))
		timer := time.NewTimer(processCloseGrace)
		defer timer.Stop()
		select {
		case <-p.done:
			return
		case <-timer.C:
		}
		if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			p.closeErr = errors.Join(p.closeErr, fmt.Errorf("stop Codex app-server: %w", err))
		}
		<-p.done
	})
	return p.closeErr
}

func closeProcessFile(f *os.File) error {
	if err := f.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
		return err
	}
	return nil
}
