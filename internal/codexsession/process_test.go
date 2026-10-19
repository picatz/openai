package codexsession

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const processTestTimeout = 5 * time.Second

// Every process launched by this file is this test executable. These tests do
// not need or execute an installed Codex CLI, a shell, credentials, or a service.
func TestProcessHelper(t *testing.T) {
	mode := os.Getenv("CODEX_PROCESS_HELPER_MODE")
	if mode == "" {
		return
	}
	switch mode {
	case "echo":
		line, err := bufio.NewReader(os.Stdin).ReadString('\n')
		if err != nil {
			os.Exit(10)
		}
		_, _ = io.WriteString(os.Stdout, line)
	case "exit-success":
		_, _ = io.WriteString(os.Stdout, "final-frame\n")
	case "exit-failure":
		_, _ = io.WriteString(os.Stderr, "synthetic-secret-not-for-errors\n")
		os.Exit(23)
	case "stall":
		_, _ = io.WriteString(os.Stdout, "ready\n")
		time.Sleep(time.Minute)
	case "eof-first":
		_ = os.Stdout.Close()
		_, _ = io.Copy(io.Discard, os.Stdin)
	case "stderr-flood":
		chunk := []byte(strings.Repeat("synthetic-private-diagnostic ", 2048))
		for range 512 {
			if _, err := os.Stderr.Write(chunk); err != nil {
				os.Exit(11)
			}
		}
		_, _ = io.WriteString(os.Stdout, "flood-complete\n")
	case "start-pipe-holder":
		executable, err := os.Executable()
		if err != nil {
			os.Exit(12)
		}
		child := exec.Command(executable, "-test.run=^TestProcessHelper$")
		child.Env = []string{
			"CODEX_PROCESS_HELPER_MODE=pipe-holder",
			"CODEX_PROCESS_HELPER_RELEASE=" + os.Getenv("CODEX_PROCESS_HELPER_RELEASE"),
			"GORACE=atexit_sleep_ms=0",
		}
		child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
		if err := child.Start(); err != nil {
			os.Exit(13)
		}
		_ = child.Process.Release()
	case "pipe-holder":
		release := os.Getenv("CODEX_PROCESS_HELPER_RELEASE")
		_ = os.WriteFile(release+".ready", []byte("ready"), 0600)
		// A self-expiry keeps a failed test from leaving a running descendant.
		deadline := time.Now().Add(2 * processTestTimeout)
		for time.Now().Before(deadline) {
			if _, err := os.Stat(release); err == nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		_ = os.Stdin.Close()
		_ = os.Stdout.Close()
		_ = os.Stderr.Close()
		_ = os.WriteFile(release+".done", []byte("done"), 0600)
	default:
		os.Exit(14)
	}
	os.Exit(0)
}

func fakeProcessCommand(t *testing.T, mode string) *exec.Cmd {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(executable, "-test.run=^TestProcessHelper$")
	cmd.Dir = t.TempDir()
	cmd.Env = []string{"CODEX_PROCESS_HELPER_MODE=" + mode, "GORACE=atexit_sleep_ms=0"}
	return cmd
}

func startFakeProcess(t *testing.T, ctx context.Context, mode string) *processPipe {
	t.Helper()
	p, err := startProcessCommand(ctx, fakeProcessCommand(t, mode))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

func waitProcessResult(t *testing.T, p *processPipe) error {
	t.Helper()
	select {
	case <-p.Done():
		return p.Wait()
	case <-time.After(processTestTimeout):
		t.Fatal("direct child was not reaped promptly")
		return nil
	}
}

func readProcessLine(t *testing.T, p *processPipe, want string) {
	t.Helper()
	result := make(chan error, 1)
	go func() {
		got, err := bufio.NewReader(p).ReadString('\n')
		if err == nil && got != want {
			err = fmt.Errorf("got line %q; want %q", got, want)
		}
		result <- err
	}()
	select {
	case err := <-result:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(processTestTimeout):
		t.Fatal("timed out reading fake child")
	}
}

func TestProcessNormalExitAndBufferedOutput(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "exit-success")
	if err := waitProcessResult(t, p); err != nil {
		t.Fatal(err)
	}
	// Wait must not discard a final frame that has not yet been consumed.
	got, err := io.ReadAll(p)
	if err != nil || string(got) != "final-frame\n" {
		t.Fatalf("buffered stdout = %q, %v", got, err)
	}
	if p.cmd.ProcessState == nil || !p.cmd.ProcessState.Success() {
		t.Fatal("Wait returned before the direct child was reaped")
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Read(make([]byte, 1)); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("read after Close = %v", err)
	}
	if _, err := p.Write([]byte("late")); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("write after Close = %v", err)
	}
}

func TestProcessReadWriteRoundTrip(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "echo")
	if _, err := io.WriteString(p, "synthetic-request\n"); err != nil {
		t.Fatal(err)
	}
	readProcessLine(t, p, "synthetic-request\n")
	if err := waitProcessResult(t, p); err != nil {
		t.Fatal(err)
	}
}

func TestProcessWaitPreservesExitError(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "exit-failure")
	first := waitProcessResult(t, p)
	var exitErr *exec.ExitError
	if !errors.As(first, &exitErr) || exitErr.ExitCode() != 23 {
		t.Fatalf("Wait = %v; want exit 23", first)
	}
	if len(exitErr.Stderr) != 0 || strings.Contains(first.Error(), "synthetic-secret") {
		t.Fatal("stderr was retained in the exit error")
	}
	if err := p.Close(); err != nil {
		t.Fatalf("Close should report cleanup separately: %v", err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			if got := p.Wait(); got != first {
				t.Errorf("Wait result changed: %v", got)
			}
		})
	}
	wg.Wait()
}

func TestProcessCloseUnblocksReadAndWrite(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "stall")
	readProcessLine(t, p, "ready\n")
	readDone, writeDone := make(chan error, 1), make(chan error, 1)
	go func() { _, err := p.Read(make([]byte, 1)); readDone <- err }()
	go func() { _, err := p.Write(make([]byte, 8<<20)); writeDone <- err }()
	// The child never reads, and never writes after its ready marker.
	select {
	case err := <-writeDone:
		t.Fatalf("large write did not block: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	closeDone := make(chan error, 1)
	go func() { closeDone <- p.Close() }()
	for name, result := range map[string]<-chan error{"read": readDone, "write": writeDone} {
		select {
		case err := <-result:
			if !errors.Is(err, os.ErrClosed) {
				t.Errorf("blocked %s = %v; want file closed", name, err)
			}
		case <-time.After(processTestTimeout):
			t.Fatalf("Close did not unblock %s", name)
		}
	}
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(processTestTimeout):
		t.Fatal("Close did not kill and join the direct child")
	}
	if err := p.Wait(); err == nil {
		t.Fatal("stalled direct child should have been killed")
	}
}

func TestProcessConcurrentRepeatedClose(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "stall")
	readProcessLine(t, p, "ready\n")
	const callers = 32
	results := make(chan error, callers)
	for range callers {
		go func() { results <- p.Close() }()
	}
	for range callers {
		select {
		case err := <-results:
			if err != nil {
				t.Error(err)
			}
		case <-time.After(processTestTimeout):
			t.Fatal("concurrent Close blocked")
		}
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if p.cmd.ProcessState == nil {
		t.Fatal("Close did not join the child")
	}
}

func TestProcessLifetimeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := startFakeProcess(t, ctx, "stall")
	readProcessLine(t, p, "ready\n")
	cancel()
	if err := waitProcessResult(t, p); err == nil {
		t.Fatal("lifetime cancellation should kill stalled direct child")
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Write([]byte("late")); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("cancellation did not close stdin: %v", err)
	}
}

func TestProcessLifetimeCancellationAfterExitClosesTransport(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := startFakeProcess(t, ctx, "exit-success")
	if err := waitProcessResult(t, p); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case <-p.closed:
	case <-time.After(processTestTimeout):
		t.Fatal("lifetime cancellation after child exit did not close the transport")
	}
	if _, err := p.Read(make([]byte, 1)); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("stdout after lifetime cancellation = %v", err)
	}
	if err := p.Wait(); err != nil {
		t.Fatalf("cancellation replaced the child's successful exit result: %v", err)
	}
}

func TestProcessCancellationBeforeStart(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cmd := fakeProcessCommand(t, "stall")
	p, err := startProcessCommand(ctx, cmd)
	if p != nil || !errors.Is(err, context.Canceled) || cmd.Process != nil {
		t.Fatalf("canceled start = %v, %v, process=%v", p, err, cmd.Process)
	}
}

func TestProcessStartupFailureClosesAssignedFiles(t *testing.T) {
	cmd := fakeProcessCommand(t, "stall")
	cmd.Dir = filepath.Join(t.TempDir(), "missing-directory")
	p, err := startProcessCommand(context.Background(), cmd)
	if p != nil || err == nil || !strings.Contains(err.Error(), "working directory") {
		t.Fatalf("failed start = %v, %v", p, err)
	}
	if cmd.Process != nil {
		t.Fatal("startup failure unexpectedly launched a child")
	}
	for name, stream := range map[string]any{"stdin": cmd.Stdin, "stdout": cmd.Stdout, "stderr": cmd.Stderr} {
		file, ok := stream.(*os.File)
		if !ok {
			t.Fatalf("%s was not a direct file", name)
		}
		// Stat on a closed Windows handle reports ERROR_INVALID_HANDLE.
		// A second Close consistently identifies an already-closed os.File.
		if err := file.Close(); !errors.Is(err, os.ErrClosed) {
			t.Errorf("failed start retained %s: %v", name, err)
		}
	}
}

func TestProcessMissingExecutable(t *testing.T) {
	// An empty temporary PATH guarantees that this never launches real Codex.
	t.Setenv("PATH", t.TempDir())
	t.Chdir(t.TempDir())
	p, err := startProcess(context.Background(), t.TempDir())
	if p != nil || err == nil || !strings.Contains(err.Error(), "install Codex") || !errors.Is(err, exec.ErrNotFound) {
		t.Fatalf("missing executable = %v, %v", p, err)
	}
}

func TestProcessRejectsInvalidCommand(t *testing.T) {
	if _, err := startProcessCommand(context.Background(), nil); err == nil {
		t.Fatal("nil command accepted")
	}
	if _, err := startProcess(nil, ""); err == nil {
		t.Fatal("nil lifetime context accepted")
	}
	cmd := fakeProcessCommand(t, "stall")
	cmd.Stderr = io.Discard
	if _, err := startProcessCommand(context.Background(), cmd); err == nil || cmd.Process != nil {
		t.Fatal("preconfigured stream accepted")
	}
}

func TestProcessStderrFloodIsDirectlyDiscarded(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "stderr-flood")
	stderr, ok := p.cmd.Stderr.(*os.File)
	if !ok || stderr.Name() != os.DevNull {
		t.Fatal("stderr is not a direct discard file")
	}
	readProcessLine(t, p, "flood-complete\n")
	if err := waitProcessResult(t, p); err != nil {
		t.Fatal(err)
	}
}

func TestProcessStdoutEOFPrecedesExit(t *testing.T) {
	p := startFakeProcess(t, context.Background(), "eof-first")
	readDone := make(chan error, 1)
	go func() { _, err := p.Read(make([]byte, 1)); readDone <- err }()
	select {
	case err := <-readDone:
		if err != io.EOF {
			t.Fatalf("Read = %v; want EOF", err)
		}
	case <-time.After(processTestTimeout):
		t.Fatal("stdout did not reach EOF")
	}
	select {
	case <-p.Done():
		t.Fatal("stdout EOF incorrectly treated as direct-child exit")
	default:
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if err := p.Wait(); err != nil {
		t.Fatalf("child did not exit on stdin EOF during grace: %v", err)
	}
}

func TestProcessInheritedPipesDoNotDelayWaitOrClose(t *testing.T) {
	release := filepath.Join(t.TempDir(), "release")
	t.Cleanup(func() { _ = os.WriteFile(release, []byte("release"), 0600) })
	cmd := fakeProcessCommand(t, "start-pipe-holder")
	cmd.Env = append(cmd.Env, "CODEX_PROCESS_HELPER_RELEASE="+release)
	p, err := startProcessCommand(context.Background(), cmd)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close() })
	waitProcessFile(t, release+".ready")
	// The descendant is deliberately alive with inherited stdin/stdout/stderr.
	// Direct-child Wait must not depend on that descendant closing its files.
	if err := waitProcessResult(t, p); err != nil {
		t.Fatal(err)
	}
	readDone := make(chan error, 1)
	go func() { _, err := p.Read(make([]byte, 1)); readDone <- err }()
	select {
	case err := <-readDone:
		t.Fatalf("inherited stdout unexpectedly closed: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	closeDone := make(chan error, 1)
	go func() { closeDone <- p.Close() }()
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(processTestTimeout):
		t.Fatal("Close waited on inherited pipes")
	}
	select {
	case err := <-readDone:
		if !errors.Is(err, os.ErrClosed) {
			t.Fatalf("inherited-pipe read after Close = %v", err)
		}
	case <-time.After(processTestTimeout):
		t.Fatal("Close left a read blocked on inherited stdout")
	}
	if err := os.WriteFile(release, []byte("release"), 0600); err != nil {
		t.Fatal(err)
	}
	waitProcessFile(t, release+".done")
}

func waitProcessFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.NewTimer(processTestTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		if _, err := os.Stat(path); err == nil {
			return
		}
		select {
		case <-deadline.C:
			t.Fatal("fake descendant did not signal its lifecycle state")
		case <-tick.C:
		}
	}
}
