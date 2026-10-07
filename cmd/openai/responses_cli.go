package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/picatz/openai/internal/conversation"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

const maxPromptBytes = 16 << 20

func newResponsesCommand(app *application) *cobra.Command {
	cmd := &cobra.Command{Use: "responses [prompt]", Args: cobra.ArbitraryArgs, Short: "Manage the OpenAI Responses API", RunE: app.runResponses}
	cmd.AddCommand(
		&cobra.Command{Use: "chat", Short: "Start an interactive Responses chat", Args: cobra.NoArgs, RunE: app.runResponses},
		&cobra.Command{Use: "create [prompt]", Short: "Create a response (reads stdin when no prompt is given)", RunE: app.createResponse},
		&cobra.Command{Use: "get <response-id>", Short: "Retrieve a stored response by ID", Args: cobra.ExactArgs(1), RunE: func(cmd *cobra.Command, args []string) error {
			resp, err := app.client.Responses.Get(cmd.Context(), args[0], responses.ResponseGetParams{})
			if err != nil {
				return fmt.Errorf("get response: %w", err)
			}
			return writeResponse(cmd.OutOrStdout(), app.output, resp)
		}},
		&cobra.Command{Use: "delete <response-id>", Short: "Delete a stored response", Args: cobra.ExactArgs(1), RunE: func(cmd *cobra.Command, args []string) error {
			if err := app.client.Responses.Delete(cmd.Context(), args[0]); err != nil {
				return fmt.Errorf("delete response: %w", err)
			}
			if app.output == "json" {
				return json.NewEncoder(cmd.OutOrStdout()).Encode(map[string]any{"id": args[0], "deleted": true})
			}
			_, err := fmt.Fprintf(cmd.OutOrStdout(), "Deleted response %q\n", args[0])
			return err
		}},
	)
	return cmd
}

func terminalFiles(cmd *cobra.Command) (*os.File, *os.File, bool) {
	in, inOK := cmd.InOrStdin().(*os.File)
	out, outOK := cmd.OutOrStdout().(*os.File)
	return in, out, inOK && outOK && term.IsTerminal(int(in.Fd())) && term.IsTerminal(int(out.Fd()))
}

func (app *application) runResponses(cmd *cobra.Command, args []string) error {
	if in, out, interactive := terminalFiles(cmd); interactive && len(args) == 0 && app.output == "text" {
		if app.legacy {
			return startResponsesChat(cmd.Context(), &app.client, app.model, in, out)
		}
		return app.runTUI(cmd, conversation.Responses)
	}
	return app.createResponse(cmd, args)
}

func readPrompt(ctx context.Context, in io.Reader, args []string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if len(args) > 0 && !(len(args) == 1 && args[0] == "-") {
		prompt := strings.Join(args, " ")
		if strings.TrimSpace(prompt) == "" {
			return "", fmt.Errorf("prompt must not be empty")
		}
		if len(prompt) > maxPromptBytes {
			return "", fmt.Errorf("prompt exceeds 16 MiB")
		}
		return prompt, nil
	}
	// Reading a pipe may block; cancellation must still let the process exit.
	type result struct {
		data []byte
		err  error
	}
	done := make(chan result, 1)
	go func() { b, err := io.ReadAll(io.LimitReader(in, maxPromptBytes+1)); done <- result{b, err} }()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case r := <-done:
		if r.err != nil {
			return "", fmt.Errorf("read prompt: %w", r.err)
		}
		if len(r.data) > maxPromptBytes {
			return "", fmt.Errorf("prompt exceeds 16 MiB")
		}
		if strings.TrimSpace(string(r.data)) == "" {
			return "", fmt.Errorf("prompt must not be empty; pass a prompt or pipe text on stdin")
		}
		return string(r.data), nil
	}
}

func (app *application) createResponse(cmd *cobra.Command, args []string) error {
	if app.sessionID != "" {
		return app.runSessionPrompt(cmd, args, conversation.Responses)
	}
	prompt, err := readPrompt(cmd.Context(), cmd.InOrStdin(), args)
	if err != nil {
		return err
	}
	params := responses.ResponseNewParams{
		Model: app.model, Store: openai.Bool(false),
		Input: responses.ResponseNewParamsInputUnion{OfString: openai.String(prompt)},
	}
	if app.webSearch {
		params.Tools = []responses.ToolUnionParam{responses.ToolParamOfWebSearchPreview(responses.WebSearchPreviewToolTypeWebSearchPreview)}
	}
	if app.stream {
		return streamResponse(cmd.Context(), &app.client, params, app.output, cmd.OutOrStdout())
	}
	resp, err := app.client.Responses.New(cmd.Context(), params)
	if err != nil {
		return fmt.Errorf("create response: %w", err)
	}
	if err := responseError(resp); err != nil {
		return err
	}
	return writeResponse(cmd.OutOrStdout(), app.output, resp)
}

func responseError(resp *responses.Response) error {
	switch resp.Status {
	case "completed":
		return nil
	case "failed":
		return fmt.Errorf("response failed: %s", resp.Error.Message)
	case "incomplete":
		return fmt.Errorf("response incomplete: %s", resp.IncompleteDetails.Reason)
	default:
		return fmt.Errorf("response did not complete (status %q)", resp.Status)
	}
}

func writeResponse(out io.Writer, format string, resp *responses.Response) error {
	if format == "json" {
		// Preserve the API payload, including newly added fields unknown to this SDK.
		return json.NewEncoder(out).Encode(json.RawMessage(resp.RawJSON()))
	}
	_, err := fmt.Fprintln(out, responseText(resp))
	return err
}

func streamResponse(ctx context.Context, client *openai.Client, params responses.ResponseNewParams, format string, out io.Writer) error {
	stream := client.Responses.NewStreaming(ctx, params)
	defer stream.Close()
	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "response.output_text.delta", "response.refusal.delta":
			if format == "text" {
				if _, err := io.WriteString(out, event.Delta); err != nil {
					return err
				}
			}
		case "response.completed":
			if err := responseError(&event.Response); err != nil {
				return err
			}
			if format == "json" {
				return writeResponse(out, format, &event.Response)
			}
			_, err := fmt.Fprintln(out)
			return err
		case "response.failed", "response.incomplete":
			return responseError(&event.Response)
		case "error":
			return fmt.Errorf("response stream: %s", event.Message)
		}
	}
	if err := stream.Err(); err != nil {
		return fmt.Errorf("response stream: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return fmt.Errorf("response stream ended before completion")
}

// responseText includes refusals, matching the streamed text behavior.
func responseText(resp *responses.Response) string {
	var text strings.Builder
	for _, item := range resp.Output {
		if item.Type != "message" {
			continue
		}
		for _, part := range item.Content {
			switch part.Type {
			case "output_text":
				text.WriteString(part.Text)
			case "refusal":
				text.WriteString(part.Refusal)
			}
		}
	}
	return text.String()
}
