package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/openai/openai-go/v3"
	"github.com/picatz/openai/internal/conversation"
	"github.com/picatz/openai/internal/tui"
	"github.com/spf13/cobra"
)

func defaultHistoryDir() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "openai", "sessions")
}
func (app *application) endpoint() string {
	if app.baseURL == "" {
		return "https://api.openai.com/v1/"
	}
	return strings.TrimRight(app.baseURL, "/") + "/"
}
func (app *application) backend() sdkBackend {
	return sdkBackend{client: &app.client, preserveReasoning: app.endpoint() == "https://api.openai.com/v1/"}
}
func (app *application) session(api conversation.API) (conversation.Session, *conversation.Store, error) {
	session := conversation.New(api, app.endpoint(), app.model)
	if app.temporary {
		if app.sessionID != "" {
			return session, nil, fmt.Errorf("--temporary and --session cannot be combined")
		}
		return session, nil, nil
	}
	if app.historyDir == "" {
		return session, nil, fmt.Errorf("cannot find a history directory; use --history-dir or --temporary")
	}
	store := &conversation.Store{Dir: app.historyDir}
	if app.sessionID != "" && app.sessionID != "new" {
		loaded, err := store.Load(app.sessionID)
		if err != nil {
			return session, nil, err
		}
		if loaded.API != api || loaded.Endpoint != app.endpoint() {
			return session, nil, fmt.Errorf("session belongs to another API or endpoint; use its original configuration or start a new session")
		}
		if loaded.Model != app.model {
			return session, nil, fmt.Errorf("session uses model %q; select it with --model or start a new session", loaded.Model)
		}
		session = loaded
	}
	return session, store, nil
}
func (app *application) runTUI(cmd *cobra.Command, api conversation.API) error {
	in, out, ok := terminalFiles(cmd)
	if !ok {
		return fmt.Errorf("interactive chat requires terminal input and output; pass a prompt or pipe text instead")
	}
	session, store, err := app.session(api)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithCancel(cmd.Context())
	defer cancel()
	err = tui.Run(tui.Config{Context: ctx, Backend: app.backend(), Session: session, Store: store, WebSearch: app.webSearch}, tea.WithInput(in), tea.WithOutput(out), tea.WithoutSignalHandler())
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}
func newSessionsCommand(app *application) *cobra.Command {
	cmd := &cobra.Command{Use: "sessions", Short: "Inspect local conversation history"}
	cmd.AddCommand(&cobra.Command{Use: "list", Short: "List local sessions", Args: cobra.NoArgs, RunE: func(cmd *cobra.Command, args []string) error {
		sessions, err := (conversation.Store{Dir: app.historyDir}).List()
		if err != nil {
			return err
		}
		type summary struct {
			ID       string           `json:"id"`
			API      conversation.API `json:"api"`
			Model    string           `json:"model"`
			Title    string           `json:"title"`
			Endpoint string           `json:"endpoint"`
		}
		rows := make([]summary, 0, len(sessions))
		for _, s := range sessions {
			rows = append(rows, summary{s.ID, s.API, s.Model, s.Title(), s.Endpoint})
		}
		if app.output == "json" {
			return json.NewEncoder(cmd.OutOrStdout()).Encode(rows)
		}
		for _, s := range rows {
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "%s\t%s\t%s\t%s\n", s.ID, s.API, s.Model, strings.ReplaceAll(s.Title, "\n", " ")); err != nil {
				return err
			}
		}
		return nil
	}}, &cobra.Command{Use: "show <id>", Short: "Show a saved local conversation", Args: cobra.ExactArgs(1), RunE: func(cmd *cobra.Command, args []string) error {
		s, err := (conversation.Store{Dir: app.historyDir}).Load(args[0])
		if err != nil {
			return err
		}
		if app.output == "json" {
			return json.NewEncoder(cmd.OutOrStdout()).Encode(s)
		}
		for _, m := range s.Messages {
			if _, err := fmt.Fprintf(cmd.OutOrStdout(), "%s: %s\n\n", m.Role, m.Content); err != nil {
				return err
			}
		}
		return nil
	}})
	return cmd
}

// runSessionPrompt shares the same transport and durable history as the TUI.
// A failed/canceled request never modifies the saved conversation.
func (app *application) runSessionPrompt(cmd *cobra.Command, args []string, api conversation.API) error {
	prompt, err := readPrompt(cmd.Context(), cmd.InOrStdin(), args)
	if err != nil {
		return err
	}
	session, store, err := app.session(api)
	if err != nil {
		return err
	}
	messages := append(append([]conversation.Message(nil), session.Messages...), conversation.Message{Role: "user", Content: prompt})
	result, err := app.backend().Generate(cmd.Context(), conversation.Request{API: api, Model: session.Model, Messages: messages, ResponsesItems: session.ResponsesItems, WebSearch: app.webSearch}, func(delta string) error {
		if app.stream && app.output == "text" {
			_, err := io.WriteString(cmd.OutOrStdout(), delta)
			return err
		}
		return nil
	})
	if err != nil {
		return err
	}
	session.Messages = append(messages, conversation.Message{Role: "assistant", Content: result.Text})
	session.ResponsesItems = result.ResponsesItems
	session.LastResponseID = result.ResponseID
	session.Usage.Input += result.Usage.Input
	session.Usage.Output += result.Usage.Output
	session.Usage.Total += result.Usage.Total
	var saveErr error
	if store != nil {
		if err := store.Save(&session); err != nil {
			saveErr = fmt.Errorf("reply received but local history was not saved: %w", err)
		}
	}

	if app.output == "json" {
		err := json.NewEncoder(cmd.OutOrStdout()).Encode(struct {
			SessionID  string             `json:"session_id"`
			ResponseID string             `json:"response_id"`
			Text       string             `json:"text"`
			Usage      conversation.Usage `json:"usage"`
			Saved      bool               `json:"saved"`
		}{session.ID, result.ResponseID, result.Text, result.Usage, store != nil && saveErr == nil})
		return errors.Join(saveErr, err)
	}

	if app.stream {
		_, err = fmt.Fprintln(cmd.OutOrStdout())
	} else {
		_, err = fmt.Fprintln(cmd.OutOrStdout(), result.Text)
	}
	if err != nil {
		return err
	}
	if store != nil && saveErr == nil {
		_, err = fmt.Fprintln(cmd.ErrOrStderr(), "Session:", session.ID)
	}
	return errors.Join(saveErr, err)
}

func (app *application) createChat(cmd *cobra.Command, args []string) error {
	if app.sessionID != "" {
		return app.runSessionPrompt(cmd, args, conversation.Chat)
	}
	prompt, err := readPrompt(cmd.Context(), cmd.InOrStdin(), args)
	if err != nil {
		return err
	}
	if app.webSearch {
		return errors.New("built-in web search is not supported by Chat Completions")
	}
	if app.stream {
		result, err := app.backend().Generate(cmd.Context(), conversation.Request{API: conversation.Chat, Model: app.model, Messages: []conversation.Message{{Role: "user", Content: prompt}}}, func(delta string) error {
			if app.output == "text" {
				_, err := io.WriteString(cmd.OutOrStdout(), delta)
				return err
			}
			return nil
		})
		if err != nil {
			return err
		}
		if app.output == "json" {
			return json.NewEncoder(cmd.OutOrStdout()).Encode(struct {
				ID    string             `json:"id"`
				Text  string             `json:"text"`
				Usage conversation.Usage `json:"usage"`
			}{result.ResponseID, result.Text, result.Usage})
		}
		_, err = fmt.Fprintln(cmd.OutOrStdout())
		return err
	}
	result, err := app.client.Chat.Completions.New(cmd.Context(), openai.ChatCompletionNewParams{Model: app.model, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage(prompt)}})
	if err != nil {
		return err
	}
	if len(result.Choices) != 1 {
		return fmt.Errorf("expected one completion choice, got %d", len(result.Choices))
	}
	choice := result.Choices[0]
	if choice.FinishReason != "stop" {
		return fmt.Errorf("chat completion did not finish normally: %s", choice.FinishReason)
	}
	if app.output == "json" {
		return json.NewEncoder(cmd.OutOrStdout()).Encode(json.RawMessage(result.RawJSON()))
	}
	_, err = fmt.Fprintln(cmd.OutOrStdout(), choice.Message.Content+choice.Message.Refusal)
	return err
}
