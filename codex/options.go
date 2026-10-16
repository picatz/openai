package codex

// Options configure a Codex client.
type Options struct {
	// CodexPathOverride points to a specific codex binary. When empty the SDK searches
	// PATH. This module does not bundle or install a Codex binary.
	CodexPathOverride string
	// BaseURL overrides the default API base URL used by the codex CLI. When empty,
	// the CLI's default value is used.
	BaseURL string
	// APIKey overrides the API key used by the codex CLI. When empty, the CLI falls
	// back to the CODEX_API_KEY environment variable.
	APIKey string
}

// ApprovalMode mirrors the codex CLI approval modes.
type ApprovalMode string

const (
	ApprovalModeNever     ApprovalMode = "never"
	ApprovalModeOnRequest ApprovalMode = "on-request"
	ApprovalModeOnFailure ApprovalMode = "on-failure"
	ApprovalModeUntrusted ApprovalMode = "untrusted"
)

// SandboxMode mirrors the codex CLI sandboxing options.
type SandboxMode string

const (
	SandboxModeReadOnly         SandboxMode = "read-only"
	SandboxModeWorkspaceWrite   SandboxMode = "workspace-write"
	SandboxModeDangerFullAccess SandboxMode = "danger-full-access"
)

// WebSearchMode controls whether Codex can use cached or live web search.
type WebSearchMode string

const (
	WebSearchModeDisabled WebSearchMode = "disabled"
	WebSearchModeCached   WebSearchMode = "cached"
	WebSearchModeLive     WebSearchMode = "live"
)

// ThreadOptions configure how a thread interacts with the codex CLI once created.
type ThreadOptions struct {
	// Model selects the model identifier to run the agent with.
	Model string
	// SandboxMode controls the filesystem sandbox granted to the agent.
	SandboxMode SandboxMode
	// WorkingDirectory sets the directory provided to --cd when launching the CLI.
	WorkingDirectory string
	// SkipGitRepoCheck mirrors --skip-git-repo-check on the CLI.
	SkipGitRepoCheck bool
	// ApprovalPolicy configures Codex's approval policy. An empty value keeps its default.
	ApprovalPolicy ApprovalMode
	// ModelReasoningEffort passes through the CLI's model_reasoning_effort setting.
	ModelReasoningEffort  string
	AdditionalDirectories []string
	NetworkAccessEnabled  *bool
	WebSearchMode         WebSearchMode
	// ConfigOverrides are repeatable KEY=VALUE TOML overrides; typed options take precedence.
	ConfigOverrides []string
}

// TurnOptions configure a single turn when running the agent.
type TurnOptions struct {
	// OutputSchema describes the expected JSON structure when requesting structured output.
	// The value must marshal to a JSON object; validation occurs before each run.
	OutputSchema any
}
