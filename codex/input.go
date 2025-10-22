package codex

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Input represents the user-provided content for a single agent turn.
// Use TextInput to send a plain string prompt, or ComposeInput with individual
// parts when mixing text and local images.
type Input struct {
	// Prompt is the base textual prompt sent to the CLI.
	Prompt string
	// Parts augments the prompt with additional inputs such as local images.
	Parts []UserInput
}

// TextInput creates an Input containing a single text prompt.
func TextInput(prompt string) Input {
	return Input{Prompt: prompt}
}

// ComposeInput creates an Input from a set of user inputs. Parts are copied so
// the caller can reuse the provided slice.
func ComposeInput(parts ...UserInput) Input {
	cp := make([]UserInput, len(parts))
	copy(cp, parts)
	return Input{Parts: cp}
}

// UserInput captures an individual segment of user supplied input.
type UserInput struct {
	// Type differentiates the payload stored in the other fields.
	Type InputType `json:"type"`
	// Text contains the textual prompt for InputTypeText entries.
	Text string `json:"text,omitempty"`
	// Path contains the local filesystem path for InputTypeLocalImage entries.
	Path string `json:"path,omitempty"`
}

// InputType enumerates the supported user input kinds.
type InputType string

const (
	InputTypeText       InputType = "text"
	InputTypeLocalImage InputType = "local_image"
)

// TextPart constructs a textual user input segment.
func TextPart(text string) UserInput {
	return UserInput{Type: InputTypeText, Text: text}
}

// LocalImagePart constructs a local image user input segment.
func LocalImagePart(path string) UserInput {
	return UserInput{Type: InputTypeLocalImage, Path: path}
}

func normalizeInput(input Input) (string, []string, error) {
	if len(input.Parts) == 0 {
		return input.Prompt, nil, nil
	}

	var promptParts []string
	if input.Prompt != "" {
		promptParts = append(promptParts, input.Prompt)
	}

	var images []string
	for idx, part := range input.Parts {
		switch part.Type {
		case InputTypeText:
			promptParts = append(promptParts, part.Text)
		case InputTypeLocalImage:
			if part.Path == "" {
				return "", nil, fmt.Errorf("input part %d: local image path must be set", idx)
			}
			images = append(images, part.Path)
		case "":
			return "", nil, fmt.Errorf("input part %d: type must be set", idx)
		default:
			return "", nil, fmt.Errorf("input part %d: unsupported type %q", idx, part.Type)
		}
	}

	prompt := strings.Join(promptParts, "\n\n")
	return prompt, images, nil
}

func ensureStructuredOutputIsJSONObject(value any) error {
	if value == nil {
		return nil
	}

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}

	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return errors.New("output schema must marshal to a JSON object")
	}
	if trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return errors.New("output schema must marshal to a JSON object")
	}
	return nil
}
