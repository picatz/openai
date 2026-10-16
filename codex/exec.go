package codex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	internalOriginatorEnv = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
	goSDKOriginator       = "codex_sdk_go"
)

// Args configure one non-interactive codex exec invocation. Prompts go to stdin,
// never through a shell. Zero-valued options retain the CLI defaults.
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
	// ConfigOverrides are repeatable KEY=VALUE TOML overrides, not file paths.
	ConfigOverrides []string
	// Deprecated: ConfigFile is a single KEY=VALUE override, despite its name.
	ConfigFile string
	// Deprecated: legacy CLI passthrough. Prefer explicit ApprovalPolicy and SandboxMode.
	FullAuto bool
	// Deprecated: legacy CLI passthrough; support depends on the installed Codex version.
	IncludePlanTool       bool
	ApprovalPolicy        ApprovalMode
	ModelReasoningEffort  string
	AdditionalDirectories []string
	// NetworkAccessEnabled is nil to keep the CLI default, or an explicit true/false.
	NetworkAccessEnabled *bool
	WebSearchMode        WebSearchMode
	Ephemeral            bool
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
	cancel    context.CancelFunc
}

func (s *ExecStream) Stdout() io.ReadCloser {
	return s.stdout
}

// Wait reaps the child and returns its exit error. Drain Stdout first, or use
// Close to stop a run whose output you no longer want. Wait is safe to repeat.
func (s *ExecStream) Wait() error {
	s.waitOnce.Do(func() {
		if s.waitFn != nil {
			s.waitErr = s.waitFn()
		}
	})
	return s.waitErr
}

// Close stops the child, closes stdout, and waits for process cleanup. It is safe
// to repeat or to call concurrently with Wait. Intentional cancellation is not
// a Close error; Wait still reports context.Canceled.
func (s *ExecStream) Close() error {
	s.closeOnce.Do(func() {
		if s.cancel != nil {
			s.cancel()
		}
		if s.stdout != nil {
			if err := s.stdout.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
				s.closeErr = err
			}
		}
		if err := s.Wait(); err != nil && !errors.Is(err, context.Canceled) {
			s.closeErr = errors.Join(s.closeErr, err)
		}
	})
	return s.closeErr
}

func (e *Exec) Run(ctx context.Context, args Args) (*ExecStream, error) {
	commandArgs, err := args.commandArgs()
	if err != nil {
		return nil, err
	}
	runCtx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(runCtx, e.path, commandArgs...)
	cmd.Env = buildEnvironment(args.BaseURL, args.APIKey)
	cmd.Stdin = strings.NewReader(args.Input)
	// Bound cleanup if a descendant retains a pipe after the CLI exits.
	cmd.WaitDelay = time.Second
	stderr := &tailBuffer{limit: 64 * 1024}
	cmd.Stderr = stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("open stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		cancel()
		_ = stdout.Close()
		return nil, fmt.Errorf("start codex exec: %w", err)
	}
	waitFn := func() error {
		err := cmd.Wait()
		// The stream's Close callback can race the parent's cancellation propagation.
		// Preserve the caller's deadline rather than reporting our cleanup cancel.
		ctxErr := ctx.Err()
		if ctxErr == nil {
			ctxErr = runCtx.Err()
		}
		cancel()
		if ctxErr != nil {
			return ctxErr
		}
		if err != nil {
			detail := strings.TrimSpace(string(stderr.data))
			if detail != "" {
				return fmt.Errorf("codex exec failed: %w: %s", err, detail)
			}
			return fmt.Errorf("codex exec failed: %w", err)
		}
		return nil
	}
	return &ExecStream{stdout: stdout, waitFn: waitFn, cancel: cancel}, nil
}

func (args Args) commandArgs() ([]string, error) {
	commandArgs := []string{"exec", "--json"}
	for _, override := range append([]string{args.ConfigFile}, args.ConfigOverrides...) {
		if override == "" {
			continue
		}
		key, _, ok := strings.Cut(override, "=")
		if !ok || strings.TrimSpace(key) == "" {
			return nil, fmt.Errorf("config override must be KEY=VALUE, not a file path: %q", override)
		}
		commandArgs = append(commandArgs, "--config", override)
	}
	addConfigString := func(key, value string) {
		if value != "" {
			// JSON string escaping is valid for TOML basic strings.
			quoted, _ := json.Marshal(value)
			commandArgs = append(commandArgs, "--config", key+"="+string(quoted))
		}
	}
	addConfigString("openai_base_url", args.BaseURL)
	addConfigString("model_reasoning_effort", args.ModelReasoningEffort)
	addConfigString("approval_policy", string(args.ApprovalPolicy))
	addConfigString("web_search", string(args.WebSearchMode))
	if args.NetworkAccessEnabled != nil {
		commandArgs = append(commandArgs, "--config", fmt.Sprintf("sandbox_workspace_write.network_access=%t", *args.NetworkAccessEnabled))
	}
	if args.Model != "" {
		commandArgs = append(commandArgs, "--model", args.Model)
	}
	if args.SandboxMode != "" {
		commandArgs = append(commandArgs, "--sandbox", string(args.SandboxMode))
	}
	if args.WorkingDirectory != "" {
		commandArgs = append(commandArgs, "--cd", args.WorkingDirectory)
	}
	for _, dir := range args.AdditionalDirectories {
		commandArgs = append(commandArgs, "--add-dir", dir)
	}
	if args.SkipGitRepoCheck {
		commandArgs = append(commandArgs, "--skip-git-repo-check")
	}
	if args.Ephemeral {
		commandArgs = append(commandArgs, "--ephemeral")
	}
	if args.OutputSchemaFile != "" {
		commandArgs = append(commandArgs, "--output-schema", args.OutputSchemaFile)
	}
	if args.OutputLastMessage != "" {
		commandArgs = append(commandArgs, "--output-last-message", args.OutputLastMessage)
	}
	for _, feature := range args.Enable {
		if feature != "" {
			commandArgs = append(commandArgs, "--enable", feature)
		}
	}
	if args.FullAuto {
		commandArgs = append(commandArgs, "--full-auto")
	}
	if args.IncludePlanTool {
		commandArgs = append(commandArgs, "--include-plan-tool")
	}
	if args.ThreadID != "" {
		commandArgs = append(commandArgs, "resume")
	}
	// Resume has its own image arguments. Keep them after the subcommand.
	for _, image := range args.Images {
		if image != "" {
			commandArgs = append(commandArgs, "--image", image)
		}
	}
	// End option parsing before positional input. Thread IDs can come from a
	// caller or a prior thread.started event; neither may introduce CLI flags.
	commandArgs = append(commandArgs, "--")
	if args.ThreadID != "" {
		commandArgs = append(commandArgs, args.ThreadID)
	}
	// An explicit stdin marker also prevents images from consuming the prompt.
	commandArgs = append(commandArgs, "-")
	return commandArgs, nil
}

// tailBuffer bounds diagnostic memory while preserving the end of stderr.
// os/exec serializes writes; it is read only after Wait has joined its copier.
type tailBuffer struct {
	data  []byte
	limit int
}

func (b *tailBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if n >= b.limit {
		b.data = append(b.data[:0], p[n-b.limit:]...)
		return n, nil
	}
	if extra := len(b.data) + n - b.limit; extra > 0 {
		b.data = append(b.data[:0], b.data[extra:]...)
	}
	b.data = append(b.data, p...)
	return n, nil
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
