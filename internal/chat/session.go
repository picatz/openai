package chat

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/openai/openai-go"
	"golang.org/x/term"
)

// defaultCachePath defines the default location for the chat session cache,
// which is used to store conversation history.
//
// On Unix-like systems, it is set to ~/.openai-cli-chat-cache, and on Windows,
// it is set to %USERPROFILE%/.openai-cli-chat-cache, which are the common locations
// for user-specific configuration files.
var defaultCachePath = cmp.Or(os.Getenv("HOME"), os.Getenv("USERPROFILE")) + "/.openai-cli-chat-cache"

// newMessageUnion converts a slice of ChatCompletionMessage into the expected union slice.
func newMessageUnion(messages []openai.ChatCompletionMessage) []openai.ChatCompletionMessageParamUnion {
	msgUnion := make([]openai.ChatCompletionMessageParamUnion, len(messages))
	for i, m := range messages {
		msgUnion[i] = m
	}
	return msgUnion
}

// CommandFunc defines the function signature for executing a command.
type CommandFunc func(ctx context.Context, session *Session, input string)

// Command represents an abstract command with a name, a matching function, and an execution function.
type Command struct {
	// Name of the command.
	//
	// If Matches is nil, the command is executed when the input matches the name.
	Name string

	// Description of the command.
	Description string

	// Matches is a function that checks if the command matches the input.
	//
	// If Matches is nil, the command is executed when the input matches the name.
	// If Matches is not nil, the command is executed when Matches returns true.
	Matches func(input string) bool

	// Run is the function that executes the command.
	Run CommandFunc
}

// builtinCommands are the built-in commands available in the chat session,
// used for managing the conversation and session state.
var builtinCommands = []Command{
	{
		Name:        "exit",
		Description: "Exit the chat session.",
		// Exiting is a special case, used for documentation.
	},
	{
		Name:        "clear",
		Description: "Clear the terminal screen.",
		Run: func(ctx context.Context, s *Session, input string) {
			s.clearScreen()
		},
	},
	{
		Name:        "erase",
		Description: "Clear the chat history.",
		Run: func(ctx context.Context, s *Session, input string) {
			s.Messages = []openai.ChatCompletionMessage{}
			s.CurrentTokensUsed = 0
			s.OutWriter.WriteString("Chat history cleared.\n")
		},
	},
	{
		Name:        "delete",
		Description: "Delete the last message.",
		Run: func(ctx context.Context, s *Session, input string) {
			if len(s.Messages) > 0 {
				s.Messages = s.Messages[:len(s.Messages)-1]
			}
		},
	},
	{
		Name:        "copy",
		Description: "Copy the last message to the clipboard.",
		Run: func(ctx context.Context, s *Session, input string) {
			if len(s.Messages) > 0 {
				if err := writeClipboard(s.Messages[len(s.Messages)-1].Content); err != nil {
					s.OutWriter.WriteString(fmt.Sprintf("Clipboard error: %s\n", err))
				}
			}
		},
	},
	{
		Name:        "system",
		Description: "Set the system context.",
		Matches: func(input string) bool {
			return strings.HasPrefix(strings.TrimSpace(input), "system:")
		},
		Run: func(ctx context.Context, s *Session, input string) {
			systemMsg := openai.ChatCompletionMessage{
				Role:    openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleSystem),
				Content: input,
			}
			s.Messages = append(s.Messages, systemMsg)
			s.OutWriter.WriteString("System context updated.\n")
		},
	},
}

// Session encapsulates the state and behavior of a CLI chat session.
// It manages terminal I/O, conversation history, caching, and command processing.
type Session struct {
	Client               *openai.Client
	Model                string
	Messages             []openai.ChatCompletionMessage
	CurrentTokensUsed    int64
	MaxCompletionTokens  int64
	MaxContextWindowSize int64
	CachePath            string

	Terminal   *term.Terminal
	OutWriter  *bufio.Writer
	TermWidth  int
	TermHeight int
	Commands   []Command
}

// NewSession creates and initializes a new chat session.
//
// It sets the terminal to raw mode, loads any existing chat history,
// and registers the default commands.
//
// A restoration function is returned to restore the terminal state on exit.
func NewSession(ctx context.Context, client *openai.Client, model string, r io.Reader, w io.Writer) (*Session, func(), error) {
	var (
		restoreFunc     = func() {} // Default no-op restore function.
		termWidth   int = 80        // Terminal width (default 80).
		termHeight  int = 24        // Terminal height (default 24).
	)

	// If we're running in a terminal, set it to "raw" mode.
	if stdin, ok := r.(*os.File); ok {
		// Get the file descriptor (number) for the terminal.
		fd := int(stdin.Fd())

		// Set the terminal to raw mode.
		oldState, err := term.MakeRaw(fd)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to set terminal to raw mode: %w", err)
		}

		// Create a restore function to reset the terminal state on exit.
		//
		// This function should be called to restore the terminal to its original state,
		// which is important for proper cleanup on exit.
		restoreFunc = func() {
			if err := term.Restore(fd, oldState); err != nil {
				fmt.Fprintf(os.Stderr, "\nfailed to restore terminal: %s\n", err)
			}
		}

		// Get the terminal size.
		//
		// This is important for rendering output correctly in the terminal.
		termWidth, termHeight, err = term.GetSize(fd)
		if err != nil {
			restoreFunc()
			return nil, nil, fmt.Errorf("failed to get terminal size while creating new chat session: %w", err)
		}
	}

	// Combine the reader and writer into a single io.ReadWriter.
	termReadWriter := struct {
		io.Reader
		io.Writer
	}{r, w}

	// Create a new terminal instance.
	t := term.NewTerminal(termReadWriter, "")

	// Set the terminal size.
	t.SetSize(termWidth, termHeight)

	// Create a buffered writer for output.
	outWriter := bufio.NewWriter(t)

	// Create a new chat session.
	cs := &Session{
		Client:            client,
		Model:             model,
		Messages:          []openai.ChatCompletionMessage{},
		CurrentTokensUsed: 0,
		CachePath:         defaultCachePath,
		Terminal:          t,
		OutWriter:         outWriter,
		TermWidth:         termWidth,
		TermHeight:        termHeight,
		Commands:          builtinCommands,
	}

	// Set up tab-completion for common commands.
	t.AutoCompleteCallback = cs.autoComplete

	// Load any existing chat history from the cache.
	if err := cs.loadCache(); err != nil {
		restoreFunc()
		return nil, nil, fmt.Errorf("failed to load chat history: %w", err)
	}

	// Return the session and the restore function.
	return cs, restoreFunc, nil
}

// Run starts the main loop of the chat session.
func (cs *Session) Run(ctx context.Context) {
	cs.clearScreen()

	cs.OutWriter.WriteString(lipgloss.NewStyle().Bold(true).Render("Welcome to the OpenAI CLI Chat Mode!") + "\n\n")
	cs.OutWriter.WriteString(lipgloss.NewStyle().Bold(true).Render("Commands") + " " +
		lipgloss.NewStyle().Faint(true).Render("(tab complete)") + "\n\n")

	for _, cmd := range cs.Commands {
		cs.OutWriter.WriteString("- " + lipgloss.NewStyle().Faint(true).Render(cmd.Name) + ": " + cmd.Description + "\n")
	}

	cs.OutWriter.WriteString("\nUse '" + lipgloss.NewStyle().Faint(true).Render("<clipboard>") +
		"' to include clipboard content in a message.\n\n")

	cs.OutWriter.Flush()

	for {
		done, err := cs.RunOnce(ctx)
		if err != nil {
			cs.OutWriter.WriteString(fmt.Sprintf("Error: %s\n", err))
			cs.OutWriter.Flush()
			if !done {
				continue
			}
		}

		if done {
			break
		}
	}

	// Save the conversation to the cache.
	if err := cs.saveCache(); err != nil {
		cs.OutWriter.WriteString(fmt.Sprintf("Failed to save chat history: %s\n", err))
		cs.OutWriter.Flush()
	}
}

func doneWithoutError() (bool, error) {
	return true, nil
}

func nonFatalError(err error) (bool, error) {
	return false, err
}

func fatalError(err error) (bool, error) {
	return true, err
}

func ranSuccessfully() (bool, error) {
	return false, nil
}

func (cs *Session) RunOnce(ctx context.Context) (bool, error) {
	cs.OutWriter.WriteString("‣ ")
	cs.OutWriter.Flush()

	input, err := cs.Terminal.ReadLine()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return doneWithoutError()
		}
		return fatalError(fmt.Errorf("failed to read input: %w", err))
	}

	trimmed := strings.TrimSpace(input)
	if trimmed == "exit" {
		return doneWithoutError()
	}

	// Process abstracted commands; if a command is executed, skip further processing.
	if cs.processInput(ctx, ptr(input)) {
		return ranSuccessfully()
	}

	// Add the user input to the conversation history.
	cs.Messages = append(cs.Messages, openai.ChatCompletionMessage{
		Role:    openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleUser),
		Content: input,
	})

	// Send the chat request and display the bot's response.
	if err := cs.chatRequest(ctx); err != nil {
		return nonFatalError(fmt.Errorf("chat request error: %w", err))
	}

	// Summarize the conversation if necessary.
	if err := cs.maybeSummarize(ctx); err != nil {
		return nonFatalError(fmt.Errorf("summarization error: %w", err))
	}

	return ranSuccessfully()
}

// processInput iterates over the abstracted commands to see if any match the input.
// If a command matches, it is executed and the function returns true.
func (cs *Session) processInput(ctx context.Context, input *string) bool {
	if input == nil {
		return false
	}

	// Ensure the output writer is flushed after each command execution,
	// to avoid common boilerplate code that each command wants to do.
	defer cs.OutWriter.Flush()

	for _, cmd := range cs.Commands {
		switch {
		case cmd.Matches == nil:
			if strings.TrimSpace(*input) == cmd.Name {
				cmd.Run(ctx, cs, *input)
				return true
			}
		case cmd.Matches(*input):
			cmd.Run(ctx, cs, *input)
			return true
		}
	}
	// Optionally, handle clipboard token replacement if needed.
	if strings.Contains(*input, "<clipboard>") {
		clip, err := readClipboard()
		if err == nil {
			*input = strings.Replace(*input, "<clipboard>", clip, -1)
		} else {
			cs.OutWriter.WriteString(fmt.Sprintf("Clipboard read error: %s\n", err))
			cs.OutWriter.Flush()
		}
	}
	return false
}

// ptr is a helper function to create a pointer to a value, because
// we're using a pointer to process the input (in case we need to modify it).
func ptr[T any](v T) *T {
	return &v
}

// chatRequest sends the conversation to the API and displays the bot's response.
func (cs *Session) chatRequest(ctx context.Context) error {
	// cs.OutWriter.WriteString(fmt.Sprintf("Processing %d messages\n", len(cs.Messages)))
	// cs.OutWriter.Flush()

	resp, err := cs.Client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:               openai.F(cs.Model),
		Messages:            openai.F(newMessageUnion(cs.Messages)),
		MaxCompletionTokens: openai.F(cmp.Or(cs.MaxCompletionTokens, 2048)),
	})
	if err != nil {
		return fmt.Errorf("failed to create chat: %w", err)
	}

	// Render the response (renderMarkdown is assumed to be implemented elsewhere).
	rendered, err := renderMarkdown(strings.TrimRight(resp.Choices[0].Message.Content, "\n"), cs.TermWidth)
	if err != nil {
		return err
	}

	cs.OutWriter.WriteString(rendered)
	cs.OutWriter.Flush()

	// Append the bot response to the conversation history and update token count.
	cs.Messages = append(cs.Messages, resp.Choices[0].Message)
	cs.CurrentTokensUsed += resp.Usage.TotalTokens

	// cs.OutWriter.WriteString(fmt.Sprintf("Tokens used: %d\n", cs.CurrentTokensUsed))
	// cs.OutWriter.Flush()

	return nil
}

// maybeSummarize checks if the token count exceeds a threshold and, if so, generates a summary.
func (cs *Session) maybeSummarize(ctx context.Context) error {
	if cs.CurrentTokensUsed >= cmp.Or(cs.MaxContextWindowSize, 4096) {
		summary, summaryTokens, err := cs.summarize(ctx, 0)
		if err != nil {
			return err
		}

		cs.OutWriter.WriteString(lipgloss.NewStyle().Width(80).
			Background(lipgloss.Color("69")).
			Foreground(lipgloss.Color("15")).
			Padding(1, 2).Render(summary) + "\n")
		cs.OutWriter.Flush()

		cs.Messages = []openai.ChatCompletionMessage{{
			Role:    openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleSystem),
			Content: "Summary of previous messages for context: " + summary,
		}}
		cs.CurrentTokensUsed = summaryTokens

		if err := cs.saveCache(); err != nil {
			return fmt.Errorf("failed to save chat history: %w", err)
		}
		cs.OutWriter.WriteString("Chat history summarized.\n")
		cs.OutWriter.Flush()
	}
	return nil
}

// summarize generates a summary of the conversation, retrying on rate limit errors if necessary.
func (cs *Session) summarize(ctx context.Context, attempts int) (string, int64, error) {
	summaryMsgs := []openai.ChatCompletionMessage{
		{
			Role: openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleSystem),
			Content: strings.Join([]string{
				"You are an expert at summarizing conversations.",
				"Write a detailed recap of the given conversation, including all important details.",
				"The summary must be at least 100 characters long and no more than 2048 characters.",
				"Ignore irrelevant content.",
			}, " "),
		},
	}

	var b strings.Builder
	for _, m := range cs.Messages {
		if m.Role == openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleSystem) {
			continue
		}
		if m.Role == openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleUser) {
			b.WriteString("User: ")
		} else {
			b.WriteString("Bot: ")
		}
		b.WriteString(m.Content + "\n\n")
	}

	summaryMsgs = append(summaryMsgs, openai.ChatCompletionMessage{
		Role:    openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleUser),
		Content: b.String(),
	})

	attempts++
	resp, err := cs.Client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:     openai.F(cs.Model),
		Messages:  openai.F(newMessageUnion(summaryMsgs)),
		MaxTokens: openai.F(int64(2048)),
	})
	if err != nil {
		if attempts < 5 && strings.Contains(err.Error(), "unexpected status code: 429") {
			time.Sleep(5 * time.Second)
			return cs.summarize(ctx, attempts)
		}
		return "", 0, err
	}
	return resp.Choices[0].Message.Content, resp.Usage.TotalTokens, nil
}

// clearScreen clears the terminal.
func (cs *Session) clearScreen() {
	cs.OutWriter.WriteString("\033[2J") // Clear the screen.
	cs.OutWriter.WriteString("\033[H")  // Move cursor to the top-left corner (like 'clear' command).
	cs.OutWriter.Flush()                // Flush the buffer to ensure the output is displayed.
}

// loadCache loads the conversation history from the cache file, if it exists.
func (cs *Session) loadCache() error {
	f, err := os.OpenFile(cs.CachePath, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return err
	}

	if fi.Size() > 0 {
		if err := json.NewDecoder(f).Decode(&cs.Messages); err != nil {
			return err
		}
	}
	return nil
}

// saveCache writes the conversation history to the cache file.
func (cs *Session) saveCache() error {
	f, err := os.OpenFile(cs.CachePath, os.O_TRUNC|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(cs.Messages)
}

// autoComplete provides basic tab-completion for common commands.
func (cs *Session) autoComplete(line string, pos int, key rune) (string, int, bool) {
	if key == '\t' {
		commands := []string{"exit", "clear", "delete", "copy", "erase", "system:", "<clipboard>"}
		for _, cmd := range commands {
			if strings.HasPrefix(cmd, line) {
				return cmd, len(cmd), true
			}
		}
	}
	return line, pos, false
}
