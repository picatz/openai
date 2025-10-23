package codex

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
)

const (
	internalOriginatorEnv = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
	goSDKOriginator       = "codex_sdk_go"
)

type Args struct {
	Input string

	BaseURL           string
	APIKey            string
	ThreadID          string
	Images            []string
	Model             string
	SandboxMode       SandboxMode
	WorkingDirectory  string
	SkipGitRepoCheck  bool
	OutputSchemaFile  string
	OutputLastMessage string
	Enable            []string
	ConfigFile        string
	FullAuto          bool
	IncludePlanTool   bool
}

type Exec struct {
	path string
}

func NewExec(pathOverride string) (*Exec, error) {
	path := pathOverride
	if path == "" {
		var err error
		path, err = findCodexPath()
		if err != nil {
			return nil, err
		}
	}
	return &Exec{path: path}, nil
}

type ExecStream struct {
	stdout    io.ReadCloser
	waitOnce  sync.Once
	waitErr   error
	waitFn    func() error
	closeOnce sync.Once
	closeErr  error
}

func (s *ExecStream) Stdout() io.ReadCloser {
	return s.stdout
}

func (s *ExecStream) Wait() error {
	s.waitOnce.Do(func() {
		if s.waitFn != nil {
			s.waitErr = s.waitFn()
		}
	})
	return s.waitErr
}

func (s *ExecStream) Close() error {
	s.closeOnce.Do(func() {
		if s.stdout != nil {
			s.closeErr = s.stdout.Close()
		}
	})
	return s.closeErr
}

func (e *Exec) Run(ctx context.Context, args Args) (*ExecStream, error) {
	commandArgs := []string{"exec", "--json"}

	if args.Model != "" {
		commandArgs = append(commandArgs, "--model", args.Model)
	}

	if args.SandboxMode != "" {
		commandArgs = append(commandArgs, "--sandbox", string(args.SandboxMode))
	}

	if args.WorkingDirectory != "" {
		commandArgs = append(commandArgs, "--cd", args.WorkingDirectory)
	}

	if args.SkipGitRepoCheck {
		commandArgs = append(commandArgs, "--skip-git-repo-check")
	}

	if args.OutputSchemaFile != "" {
		commandArgs = append(commandArgs, "--output-schema", args.OutputSchemaFile)
	}

	if args.OutputLastMessage != "" {
		commandArgs = append(commandArgs, "--output-last-message", args.OutputLastMessage)
	}

	for _, image := range args.Images {
		if image != "" {
			commandArgs = append(commandArgs, "--image", image)
		}
	}

	for _, feature := range args.Enable {
		if feature != "" {
			commandArgs = append(commandArgs, "--enable", feature)
		}
	}

	if args.ConfigFile != "" {
		commandArgs = append(commandArgs, "--config", args.ConfigFile)
	}

	if args.FullAuto {
		commandArgs = append(commandArgs, "--full-auto")
	}

	if args.IncludePlanTool {
		commandArgs = append(commandArgs, "--include-plan-tool")
	}

	if args.ThreadID != "" {
		commandArgs = append(commandArgs, "resume", args.ThreadID)
	}

	cmd := exec.CommandContext(ctx, e.path, commandArgs...)

	env := buildEnvironment(args.BaseURL, args.APIKey)
	cmd.Env = env

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("open stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("open stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("open stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start codex exec: %w", err)
	}

	stderrBuf := bytes.NewBuffer(nil)
	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		_, _ = io.Copy(stderrBuf, stderr)
	}()

	writeErrCh := make(chan error, 1)
	go func() {
		defer stdin.Close()
		_, err := io.WriteString(stdin, args.Input)
		writeErrCh <- err
	}()

	waitFn := func() error {
		err := cmd.Wait()
		writeErr := <-writeErrCh
		<-stderrDone

		if writeErr != nil {
			return fmt.Errorf("write to codex stdin: %w", writeErr)
		}

		if err != nil {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) {
				stderrText := strings.TrimSpace(stderrBuf.String())
				if stderrText != "" {
					return fmt.Errorf("codex exec failed: %s: %s", exitErr, stderrText)
				}
				return fmt.Errorf("codex exec failed: %w", err)
			}
			return err
		}

		return nil
	}

	return &ExecStream{stdout: stdout, waitFn: waitFn}, nil
}

func buildEnvironment(baseURL, apiKey string) []string {
	envMap := make(map[string]string)
	for _, kv := range os.Environ() {
		if idx := strings.IndexByte(kv, '='); idx >= 0 {
			envMap[kv[:idx]] = kv[idx+1:]
		}
	}

	if value, ok := envMap[internalOriginatorEnv]; !ok || value == "" {
		envMap[internalOriginatorEnv] = goSDKOriginator
	}
	if baseURL != "" {
		envMap["OPENAI_BASE_URL"] = baseURL
	}
	if apiKey != "" {
		envMap["CODEX_API_KEY"] = apiKey
	}

	env := make([]string, 0, len(envMap))
	for k, v := range envMap {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	return env
}

func findCodexPath() (string, error) {
	codexPath, err := exec.LookPath("codex")
	if err != nil {
		return "", fmt.Errorf("find codex binary: %w", err)
	}
	return codexPath, nil
}

// Run executes a codex command with the specified arguments and
// returns an iterator that yields ThreadEvents from the execution stream.
//
// This is a high-level convenience function that combines creating an Exec,
// running it with the provided arguments, and streaming the resulting events.
// It handles proper cleanup of the execution stream.
func Run(ctx context.Context, args Args) iter.Seq2[*ThreadEvent, error] {
	return func(yield func(*ThreadEvent, error) bool) {
		exec, err := NewExec("")
		if err != nil {
			yield(nil, fmt.Errorf("create codex exec: %w", err))
			return
		}

		stream, err := exec.Run(ctx, args)
		if err != nil {
			yield(nil, fmt.Errorf("run codex exec: %w", err))
			return
		}
		defer stream.Close()

		for event, err := range EventStream(ctx, stream) {
			if err != nil {
				if !yield(nil, fmt.Errorf("read codex event: %w", err)) {
					return
				}
				continue
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}
