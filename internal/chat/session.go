package chat

import (
	"bufio"
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/openai/openai-go"
	"github.com/picatz/openai/internal/chat/storage"
	"github.com/segmentio/ksuid"
	"golang.org/x/term"
)

// DefaultCachePath defines the default location for the chat session cache,
// which is used to store conversation history as a [pebble]-backed database.
//
// On Unix-like systems, it is set to ~/.openai-cli-chat-cache, and on Windows,
// it is set to %USERPROFILE%/.openai-cli-chat-cache, which are the common locations
// for user-specific configuration files.
//
// [pebble]: https://github.com/cockroachdb/pebble
var DefaultCachePath = cmp.Or(os.Getenv("HOME"), os.Getenv("USERPROFILE")) + "/.openai-cli-chat-pebble-storage-cache"

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
		Name:        "erase all",
		Description: "Clear the chat history and backend storage.",
		Run: func(ctx context.Context, s *Session, input string) {
			// Prompt the user for confirmation before clearing the chat history.
			s.OutWriter.WriteString("\nAre you sure you want to clear the chat history? (y/n): ")
			s.OutWriter.Flush()

			confirmation, err := s.Terminal.ReadLine()
			if err != nil {
				s.OutWriter.WriteString(fmt.Sprintf("Error reading confirmation: %s\n", err))
				return
			}

			if strings.ToLower(strings.TrimSpace(confirmation)) != "y" {
				s.OutWriter.WriteString("\nChat history not cleared.\n")
				return
			}

			s.Messages = []openai.ChatCompletionMessage{}
			s.CurrentTokensUsed = 0

			var (
				perPage       = storage.PageSize(10)
				nextPageToken *string
			)

			for {
				entries, nextPageToken, err := s.StorageBackend.List(ctx, perPage, nextPageToken)
				if err != nil {
					s.OutWriter.WriteString(fmt.Sprintf("Error listing entries: %s\n", err))
					break
				}

				for key := range entries {
					if err := s.StorageBackend.Delete(ctx, key); err != nil {
						s.OutWriter.WriteString(fmt.Sprintf("Error deleting entry %s: %s\n", key, err))
					}
				}

				if nextPageToken == nil {
					break
				}
			}

			err = s.StorageBackend.Flush(ctx)
			if err != nil {
				s.OutWriter.WriteString(fmt.Sprintf("Error flushing backend storage: %s\n", err))
			} else {
				s.OutWriter.WriteString("\nChat history cleared in memory and backend.\n\n")
			}
		},
	},
	{
		Name:        "search",
		Description: "Search the chat history for a specific message or keyword.",
		Matches: func(input string) bool {
			return strings.HasPrefix(strings.TrimSpace(input), "search:")
		},
		Run: func(ctx context.Context, s *Session, input string) {
			query := strings.TrimSpace(strings.TrimPrefix(input, "search:"))
			if query == "" {
				s.OutWriter.WriteString("Please provide a search query.\n")
				return
			}

			// s.OutWriter.WriteString(fmt.Sprintf("Searching for: %s\n", query))
			// s.OutWriter.Flush()

			emedResp, err := s.Client.Embeddings.New(ctx, openai.EmbeddingNewParams{
				Model: openai.F(cmp.Or(os.Getenv("OPENAI_EMBEDDING_MODEL"), openai.EmbeddingModelTextEmbedding3Large)),
				Input: openai.F(openai.EmbeddingNewParamsInputUnion(openai.EmbeddingNewParamsInputArrayOfStrings{query})),
			})
			if err != nil {
				s.OutWriter.WriteString(fmt.Sprintf("Error creating embedding: %s\n", err))
				return
			}

			// Search the cache for matching entries.
			results := s.searchCache(ctx, 10, func(key string, pair ReqRespPair) bool {
				return MatchQueryCosignSimilarity(s, pair.EmbeddingModel, emedResp.Data[0].Embedding, pair)
			})

			var numMatches int

			for key, value := range results {
				s.OutWriter.WriteString(fmt.Sprintf("\t%s (%s): %s\n\n", value.Req.Role, key, value.Req.Content))
				s.OutWriter.WriteString(fmt.Sprintf("\t%s (%s): %s\n\n", value.Resp.Role, key, value.Resp.Content))
				s.OutWriter.WriteString(fmt.Sprintf("\tTokens used: %d\n\n", value.ReqTokens+value.RespTokens))
				s.OutWriter.WriteString("---\n")
				numMatches++
			}

			if numMatches > 0 {
				s.OutWriter.WriteString(fmt.Sprintf("\nFound %d matches.\n", numMatches))
			}
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
	{
		Name:        "help",
		Description: "Show help for commands.",
		Run: func(ctx context.Context, s *Session, input string) {
			s.ShowHelp()
		},
	},
	{
		Name:        "tokens",
		Description: "Show the number of tokens used.",
		Run: func(ctx context.Context, s *Session, input string) {
			s.OutWriter.WriteString(fmt.Sprintf("Tokens used: %d\n", s.CurrentTokensUsed))
		},
	},
	{
		Name:        "messages",
		Description: "Show the chat messages currently being used with the model.",
		Run: func(ctx context.Context, s *Session, input string) {
			for _, msg := range s.Messages {
				s.OutWriter.WriteString(fmt.Sprintf("\n\t%s: %s\n", msg.Role, msg.Content))
			}
		},
	},
	{
		Name: "history",
		Matches: func(input string) bool {
			// Matches "history" or "history <number>".
			switch {
			case strings.TrimSpace(input) == "history":
				return true
			case strings.HasPrefix(strings.TrimSpace(input), "history "):
				// Check if a number is provided.
				parts := strings.Fields(input)
				if len(parts) == 2 {
					_, err := strconv.Atoi(parts[1])
					return err == nil
				}
				return false
			default:
				return false
			}
		},
		Description: "Show the chat message history from the backend storage.",
		Run: func(ctx context.Context, s *Session, input string) {
			// Default to showing the last 10 messages if no number is provided.
			numToShow := 10
			if strings.TrimSpace(input) != "history" {
				parts := strings.Fields(input)
				if len(parts) == 2 {
					num, err := strconv.Atoi(parts[1])
					if err == nil {
						numToShow = num
					}
				}
			}

			if numToShow <= 0 {
				s.OutWriter.WriteString("Invalid number of messages to show.\n")
				return
			}

			entries, _, err := s.StorageBackend.List(ctx, storage.PageSize(numToShow), nil)
			if err != nil {
				s.OutWriter.WriteString(fmt.Sprintf("Error listing entries: %s\n", err))
				return
			}

			for key, value := range entries {
				s.OutWriter.WriteString(fmt.Sprintf("\t%s (%s): %s\n\n", value.Req.Role, key, value.Req.Content))
				s.OutWriter.WriteString(fmt.Sprintf("\t%s (%s): %s\n\n", value.Resp.Role, key, value.Resp.Content))
				s.OutWriter.WriteString(fmt.Sprintf("\tTokens used: %d\n\n", value.ReqTokens+value.RespTokens))
				s.OutWriter.WriteString("---\n")
			}
		},
	},
}

// ReqRespPair represents a request-response pair in the chat session,
// used for storing conversation history in the backend.
type ReqRespPair struct {
	Model          string                       `json:"model,omitzero"`
	Req            openai.ChatCompletionMessage `json:"req,omitzero"`
	ReqTokens      int64                        `json:"req_tokens,omitzero"`
	Resp           openai.ChatCompletionMessage `json:"resp,omitzero"`
	RespTokens     int64                        `json:"resp_tokens,omitzero"`
	EmbeddingModel string                       `json:"embedding_model,omitzero"`
	Embedding      []float64                    `json:"embedding,omitzero"`
}

// Session encapsulates the state and behavior of a CLI chat session.
// It manages terminal I/O, conversation history, caching, and command processing.
type Session struct {
	Client                     *openai.Client
	Model                      string
	StorageBackend             storage.Backend[string, ReqRespPair]
	Messages                   []openai.ChatCompletionMessage
	CurrentTokensUsed          int64
	SummarizeContextWindowSize int64

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
func NewSession(ctx context.Context, client *openai.Client, model string, r io.Reader, w io.Writer, b storage.Backend[string, ReqRespPair]) (*Session, func(), error) {
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
		StorageBackend:    b,
		Messages:          []openai.ChatCompletionMessage{},
		CurrentTokensUsed: 0,
		Terminal:          t,
		OutWriter:         outWriter,
		TermWidth:         termWidth,
		TermHeight:        termHeight,
		Commands:          builtinCommands,
	}

	// Set up tab-completion for common commands.
	t.AutoCompleteCallback = cs.autoComplete

	// Load any existing chat history from the cache.
	if err := cs.loadCache(ctx); err != nil {
		restoreFunc()
		return nil, nil, fmt.Errorf("failed to load chat history: %w", err)
	}

	// Return the session and the restore function.
	return cs, restoreFunc, nil
}

func (cs *Session) ShowHelp() {
	cs.OutWriter.WriteString(lipgloss.NewStyle().Bold(true).Render("Commands") + " " +
		lipgloss.NewStyle().Faint(true).Render("(tab complete)") + "\n\n")

	for _, cmd := range cs.Commands {
		cs.OutWriter.WriteString("- " + lipgloss.NewStyle().Faint(true).Render(cmd.Name) + ": " + cmd.Description + "\n")
	}

	cs.OutWriter.WriteString("\nUse '" + lipgloss.NewStyle().Faint(true).Render("<clipboard>") +
		"' to include clipboard content in a message.\n\n")

	cs.OutWriter.Flush()
}

// Run starts the main loop of the chat session.
func (cs *Session) Run(ctx context.Context) {
	cs.clearScreen()

	// User is new, show the welcome message.
	//
	// This is only shown once, when the user starts the session.
	if len(cs.Messages) == 0 {
		cs.OutWriter.WriteString(lipgloss.NewStyle().Bold(true).Render("Welcome to the OpenAI CLI Chat Mode!") + "\n\n")
		cs.ShowHelp()
	}

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
	if err := cs.saveCache(ctx); err != nil {
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

	nextUserMessage := openai.ChatCompletionMessage{
		Role:    openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleUser),
		Content: input,
	}

	// Send the chat request and display the bot's response, storing the conversation history.
	if err := cs.chatRequest(ctx, nextUserMessage); err != nil {
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
func (cs *Session) chatRequest(ctx context.Context, nextUserMessage openai.ChatCompletionMessage) error {
	cs.Messages = append(cs.Messages, nextUserMessage)

	resp, err := cs.Client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:    openai.F(cs.Model),
		Messages: openai.F(newMessageUnion(cs.Messages)),
		// TODO(kent): consider this more.
		//
		// MaxCompletionTokens: openai.F(cmp.Or(cs.MaxCompletionTokens, 2048)),
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

	embedResp, err := cs.Client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: openai.F(cmp.Or(os.Getenv("OPENAI_EMBEDDING_MODEL"), openai.EmbeddingModelTextEmbedding3Large)),
		Input: openai.F(
			openai.EmbeddingNewParamsInputUnion(
				openai.EmbeddingNewParamsInputArrayOfStrings{
					nextUserMessage.Content,
					resp.Choices[0].Message.Content,
				},
			),
		),
	})
	if err != nil {
		return fmt.Errorf("failed to create embedding for message: %w", err)
	}

	// The reqRespPairKey is a K-Sortable Unique IDentifier (KSUID) for the request and response.
	//
	// This is useful for iterating over the cache in a sorted order, which we can
	// use to do things like summarize the conversation based on the most recent
	// messages in the backend.
	reqRespPairKey := fmt.Sprintf("%s-%s", ksuid.New(), resp.ID)

	// Save the request and response to the backend storage.
	if err := cs.StorageBackend.Set(ctx, reqRespPairKey, ReqRespPair{
		Model:          cs.Model,
		Req:            nextUserMessage,
		ReqTokens:      resp.Usage.PromptTokens,
		Resp:           resp.Choices[0].Message,
		RespTokens:     resp.Usage.CompletionTokens,
		EmbeddingModel: cmp.Or(os.Getenv("OPENAI_EMBEDDING_MODEL"), openai.EmbeddingModelTextEmbedding3Large),
		Embedding:      embedResp.Data[0].Embedding,
	}); err != nil {
		return fmt.Errorf("failed to save chat response to backend storage: %w", err)
	}

	// cs.OutWriter.WriteString(fmt.Sprintf("Tokens used: %d\n", cs.CurrentTokensUsed))
	// cs.OutWriter.Flush()

	return nil
}

// maybeSummarize checks if the token count exceeds a threshold and, if so, generates a summary.
func (cs *Session) maybeSummarize(ctx context.Context) error {
	if cs.CurrentTokensUsed >= cmp.Or(cs.SummarizeContextWindowSize, 4096) {
		summary, summaryTokens, err := cs.summarize(ctx, 0)
		if err != nil {
			return err
		}

		cs.Messages = []openai.ChatCompletionMessage{{
			Role:    openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleSystem),
			Content: "Summary of previous messages for context: " + summary,
		}}
		cs.CurrentTokensUsed = summaryTokens

		if err := cs.saveCache(ctx); err != nil {
			return fmt.Errorf("failed to save chat history: %w", err)
		}
		cs.OutWriter.WriteString("\nChat history summarized.\n")
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
				"Ignore irrelevant content.",
			}, " "),
		},
	}

	var b strings.Builder
	for _, m := range cs.Messages {
		if m.Role == openai.ChatCompletionMessageRole(openai.ChatCompletionMessageParamRoleSystem) {
			continue
		}
		b.WriteString(string(m.Role) + ":\n" + m.Content + "\n")
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

// loadCache loads the most recent conversation history from the cache, if it exists.
func (cs *Session) loadCache(ctx context.Context) error {
	entries, _, err := cs.StorageBackend.List(ctx, storage.PageSize(10), nil)
	if err != nil {
		return fmt.Errorf("failed to list chat cache: %w", err)
	}

	for _, value := range entries {
		cs.CurrentTokensUsed += (value.ReqTokens + value.RespTokens)
		cs.Messages = append(cs.Messages, value.Req)
		cs.Messages = append(cs.Messages, value.Resp)
	}

	if err := cs.maybeSummarize(ctx); err != nil {
		return fmt.Errorf("failed to summarize chat after loading from cache: %w", err)
	}

	return nil
}

// saveCache writes the conversation history to the cache file.
func (cs *Session) saveCache(ctx context.Context) error {
	if err := cs.StorageBackend.Flush(ctx); err != nil {
		return fmt.Errorf("failed to save chat cache: %w", err)
	}
	return nil
}

func cosignSimilarity(a, b []float64) float64 {
	var (
		dotProduct float64
		magnitudeA float64
		magnitudeB float64
	)

	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}

	return dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB))
}

func MatchQueryCosignSimilarity(s *Session, embeddingModel string, query []float64, pair ReqRespPair) bool {
	if embeddingModel != pair.EmbeddingModel {
		return false
	}

	value := cosignSimilarity(query, pair.Embedding)

	// s.OutWriter.Write([]byte(fmt.Sprintf("Cosine similarity: %f\n", value)))
	// s.OutWriter.Flush()

	return value > 0.8
}

func (cs *Session) searchCache(ctx context.Context, n int, match func(key string, pair ReqRespPair) bool) iter.Seq2[string, ReqRespPair] {
	return func(yield func(string, ReqRespPair) bool) {
		var (
			perPage       = storage.PageSize(10)
			nextPageToken *string
		)

		matched := 0

		for {
			entries, nextPageToken, err := cs.StorageBackend.List(ctx, perPage, nextPageToken)
			if err != nil {
				cs.OutWriter.Write(fmt.Appendf(nil, "Error listing entries: %s\n", err))
				cs.OutWriter.Flush()
				// Return an error if the listing fails.
				return
			}

			for key, pair := range entries {
				if match(key, pair) {
					if !yield(key, pair) {
						return
					}
				}

				matched++

				if matched >= n {
					return
				}
			}

			if nextPageToken == nil {
				break
			}
		}

		if matched == 0 {
			cs.OutWriter.Write([]byte("No matches found.\n"))
			cs.OutWriter.Flush()
		}
	}
}

// autoComplete provides basic tab-completion for common commands.
func (cs *Session) autoComplete(line string, pos int, key rune) (string, int, bool) {
	if key == '\t' {
		commands := []string{}

		for _, cmd := range cs.Commands {
			commands = append(commands, cmd.Name)
		}

		for _, cmd := range commands {
			if strings.HasPrefix(cmd, line) {
				return cmd, len(cmd), true
			}
		}
	}
	return line, pos, false
}
