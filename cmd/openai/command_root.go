package main

import (
	"cmp"
	"fmt"
	"net/url"
	"os"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/spf13/cobra"
)

type application struct {
	legacy     bool
	temporary  bool
	historyDir string
	sessionID  string
	client     openai.Client
	model      string
	baseURL    string
	timeout    time.Duration
	output     string
	stream     bool
	webSearch  bool
}

// newRootCommand constructs independent command trees. SDK environment defaults
// (API key, organization, project, base URL) are shared by every operation.
func newRootCommand(options ...option.RequestOption) *cobra.Command {
	app := &application{model: cmp.Or(os.Getenv("OPENAI_MODEL"), "gpt-4o"),
		baseURL: cmp.Or(os.Getenv("OPENAI_BASE_URL"), os.Getenv("OPENAI_API_URL"))}
	root := &cobra.Command{
		Use: "openai [prompt]", Args: cobra.ArbitraryArgs, Short: "OpenAI CLI", SilenceUsage: true, SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			if app.legacy && app.sessionID != "" {
				return fmt.Errorf("--session is not supported with --legacy")
			}
			if app.temporary && app.sessionID != "" {
				return fmt.Errorf("--temporary and --session cannot be combined")
			}
			if app.timeout < 0 {
				return fmt.Errorf("timeout must not be negative")
			}
			if app.model == "" {
				return fmt.Errorf("model must not be empty")
			}
			if app.output != "text" && app.output != "json" {
				return fmt.Errorf("output must be text or json")
			}
			opts := []option.RequestOption{}
			if app.baseURL != "" {
				u, err := url.Parse(app.baseURL)
				if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
					return fmt.Errorf("base URL must be an HTTP(S) URL without credentials, query, or fragment")
				}
				opts = append(opts, option.WithBaseURL(app.baseURL))
			}
			if app.timeout > 0 {
				opts = append(opts, option.WithRequestTimeout(app.timeout))
			}
			app.client = openai.NewClient(append(opts, options...)...)
			return nil
		},
		RunE: app.runResponses,
	}
	root.PersistentFlags().StringVar(&app.model, "model", app.model, "Text model (OPENAI_MODEL)")
	root.PersistentFlags().StringVar(&app.baseURL, "base-url", app.baseURL, "API base URL (OPENAI_BASE_URL; legacy OPENAI_API_URL)")
	root.PersistentFlags().DurationVar(&app.timeout, "timeout", 2*time.Minute, "Per-request timeout (0 disables)")
	root.PersistentFlags().StringVarP(&app.output, "output", "o", "text", "Output format: text or json")
	root.PersistentFlags().BoolVar(&app.stream, "stream", false, "Stream response text; JSON emits one completed response")
	root.PersistentFlags().BoolVar(&app.webSearch, "web-search", false, "Enable the Responses web search tool")
	root.PersistentFlags().BoolVar(&app.legacy, "legacy", false, "Use the previous terminal interface and history")
	root.PersistentFlags().BoolVarP(&app.temporary, "temporary", "t", false, "Do not save local session history")
	root.PersistentFlags().StringVar(&app.historyDir, "history-dir", defaultHistoryDir(), "Local session history directory")
	root.PersistentFlags().StringVar(&app.sessionID, "session", "", "Continue a local session ID, or use new to create one")
	root.AddCommand(newResponsesCommand(app), newChatCommand(app), newImageCommand(app), newAssistantCommand(), newSessionsCommand(app))
	return root
}
