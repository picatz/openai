package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/picatz/openai/internal/responses"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

func init() {
	responsesCommand.AddCommand(
		responsesChatCommand,
		responsesGetCommand,
		responsesDeleteCommand,
	)

	rootCmd.AddCommand(
		responsesCommand,
	)
}

var responsesCommand = &cobra.Command{
	Use:   "responses",
	Short: "Manage the OpenAI Responses API",
	RunE: func(cmd *cobra.Command, args []string) error {
		client := responses.NewClient(os.Getenv("OPENAI_API_KEY"), http.DefaultClient)

		startResonsesChat(cmd.Context(), client, "gpt-4o")

		return nil
	},
}

var responsesChatCommand = &cobra.Command{
	Use:   "chat",
	Short: "Chat with the OpenAI Responses API",
	RunE: func(cmd *cobra.Command, args []string) error {
		client := responses.NewClient(os.Getenv("OPENAI_API_KEY"), http.DefaultClient)

		startResonsesChat(cmd.Context(), client, "gpt-4o")

		return nil
	},
}

var responsesGetCommand = &cobra.Command{
	Use:   "get",
	Short: "Get a single response",
	RunE: func(cmd *cobra.Command, args []string) error {
		client := responses.NewClient(os.Getenv("OPENAI_API_KEY"), http.DefaultClient)

		resp, err := client.Create(cmd.Context(), responses.Request{
			Model:      "gpt-4o",
			Input:      responses.Text(strings.Join(args, " ")),
			ToolChoice: responses.RequestToolChoiceAuto,
			Tools: responses.RequestTools{
				responses.RequestToolWebSearchPreview{},
			},
			Store: false,
		})
		if err != nil {
			return fmt.Errorf("failed to create response: %w", err)
		}

		for _, output := range resp.Output {
			if output.Type != "message" {
				continue
			}
			cmd.OutOrStdout().Write([]byte(output.Content[0].Text + "\n"))
		}

		return nil
	},
}

var responsesDeleteCommand = &cobra.Command{
	Use:   "delete",
	Short: "Delete a single response",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		client := responses.NewClient(os.Getenv("OPENAI_API_KEY"), http.DefaultClient)

		respID := args[0]
		if err := client.Delete(cmd.Context(), respID); err != nil {
			return fmt.Errorf("failed to delete response %q: %w", respID, err)
		}

		cmd.OutOrStdout().Write([]byte(fmt.Sprintf("Deleted response %q\n", respID)))
		return nil
	},
}

func startResonsesChat(ctx context.Context, client *responses.Client, model string) {
	// Set the terminal to raw mode.
	oldState, err := term.MakeRaw(0)
	if err != nil {
		panic(err)
	}
	defer term.Restore(0, oldState)

	termWidth, termHeight, err := term.GetSize(0)
	if err != nil {
		panic(err)
	}

	termReadWriter := struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout}

	t := term.NewTerminal(termReadWriter, "") // Will set the prompt later.

	t.SetSize(termWidth, termHeight)

	// Use buffered output so we can write to the terminal without
	// having to wait for a newline, and so we can clear the screen
	// and move the cursor around without having to worry about
	// overwriting the prompt.
	bt := bufio.NewWriter(t)

	cls := func() {
		// Clear the screen.
		bt.WriteString("\033[2J")

		// Move to the top left.
		bt.WriteString("\033[H")

		// Flush the buffer to the terminal.
		bt.Flush()
	}

	cls()

	// Autocomplete for commands.
	t.AutoCompleteCallback = func(line string, pos int, key rune) (newLine string, newPos int, ok bool) {
		// If the user presses tab, then autocomplete the command.
		if key == '\t' {
			for _, cmd := range []string{"exit", "clear", "delete"} {
				if strings.HasPrefix(cmd, line) {
					// Autocomplete the command.
					// t.Write([]byte(cmd[len(line):]))

					// Return the new line and position, which must come after the
					// command.
					return cmd, len(cmd), true
				}
			}
		}

		// If the user hit backspace on the example system message, then we'll
		// just delete the whole line, re-add the "system:" prefix, and return
		// the new line and position.

		// Otherwise, we'll just return the line.
		return line, pos, false
	}

	// Print welcome message.
	bt.WriteString(styleBold.Render("Welcome to the OpenAI API Responses CLI chat mode!"))
	bt.WriteString("\n\n")
	bt.WriteString(styleWarning.Render("WARNING") + styleFaint.Render(": All responses are stored and deleted after exiting.\n\n"))
	bt.WriteString("\033[0G")
	bt.WriteString(styleBold.Render("Commands") + " " + styleFaint.Render("(tab complete)") + "\n\n")
	bt.WriteString("- " + styleFaint.Render("clear") + " to clear screen.\n")
	bt.WriteString("- " + styleFaint.Render("delete") + " to delete previous response (up to given number).\n")
	bt.WriteString("- " + styleFaint.Render("exit") + " to quit.\n\n")
	bt.Flush()

	var allRespIDs []string
	defer func() {
		total := len(allRespIDs)
		if total == 0 {
			return
		}
		bt.WriteString("\n")
		bt.Flush()

		for i, respID := range allRespIDs {
			if err := client.Delete(ctx, respID); err != nil {
				bt.WriteString(respID + ":" + err.Error() + "\n")
				bt.Flush()
				return
			}
			var (
				progress      = i + 1
				percent       = float64(progress) / float64(total)
				barWidth      = 20
				completedBars = int(percent * float64(barWidth))
				remainingBars = barWidth - completedBars
				progressBar   = strings.Repeat("█", completedBars) + strings.Repeat("_", remainingBars)
			)
			bt.WriteString(styleFaint.Render("\033[0G" + fmt.Sprintf("Deleting responses %s (%d/%d)", progressBar, progress, total)))
			bt.Flush()
		}
		bt.WriteString("\n")
		bt.WriteString("\n")
		bt.Flush()
	}()

	var prevRespID string

	for {
		// Move to left edge.
		bt.WriteString("\033[0G")

		// Print the prompt.
		bt.WriteString("‣ ")

		// Flush the buffer to the terminal.
		bt.Flush()

		// Read up to line from STDIN.
		input, err := t.ReadLine()
		if err != nil {
			bt.WriteString(err.Error())
			bt.Flush()
			return
		}

		// Check if the user wants to exit.
		if strings.TrimSpace(input) == "exit" {
			break
		}

		// Check if user wants to clear the screen.
		if strings.TrimSpace(input) == "clear" {
			// Clear the screen.
			cls()
			continue
		}

		fields := strings.Fields(input)

		switch fields[0] {
		case "delete":
			if len(fields) == 1 {
				bt.WriteString("Usage: delete <number>\n")
				bt.Flush()
				continue
			}
			if len(fields) == 2 {
				num, err := strconv.Atoi(fields[1])
				if err != nil {
					bt.WriteString("Usage: delete <number>\n")
					bt.Flush()
					continue
				}
				var deletedN int
				for i := 0; i < num && len(allRespIDs) > 0; i++ {
					respID := allRespIDs[len(allRespIDs)-1]     // Get the last response ID.
					allRespIDs = allRespIDs[:len(allRespIDs)-1] // Remove it from the list.

					if err := client.Delete(ctx, respID); err != nil {
						bt.WriteString(respID + ":" + err.Error() + "\n")
						bt.Flush()
						return
					}

					if prevRespID == respID {
						prevRespID = ""
					}

					deletedN++
				}
				bt.WriteString(fmt.Sprintf("Deleted %d responses.\n", deletedN))
				bt.Flush()
				continue
			}
		}

		resp, err := client.Create(ctx, responses.Request{
			Model:              model,
			PreviousResponseID: prevRespID,
			Input:              responses.Text(input),
			ToolChoice:         responses.RequestToolChoiceAuto,
			Tools: responses.RequestTools{
				responses.RequestToolWebSearchPreview{},
			},
		})
		if err != nil {
			bt.WriteString(err.Error())
			bt.Flush()
			return
		}

		for _, output := range resp.Output {
			if output.Type != "message" {
				continue
			}

			// Get 3/4 of the terminal width.
			threeQuarterWidth := termWidth * 3 / 4

			// Print the output using markdown-friendly terminal rendering.
			s, err := renderMarkdown(strings.TrimRight(output.Content[0].Text, "\n"), threeQuarterWidth)
			if err != nil {
				bt.WriteString(err.Error())
				bt.Flush()
				return
			}

			bt.WriteString(s)
		}

		allRespIDs = append(allRespIDs, resp.ID)
		prevRespID = resp.ID
	}
}
