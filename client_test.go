package openai_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/picatz/openai"
)

func TestCreateCompletion(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := c.CreateCompletion(ctx, nil, &openai.CreateCompletionRequest{
		Model:     openai.ModelDavinci,
		Prompt:    []string{"This is a test"},
		MaxTokens: 5,
	})

	if err != nil {
		t.Fatal(err)
	}

	for i, choice := range resp.Choices {
		t.Logf("choice %d text: %v", i, choice.Text)
		t.Logf("choice %d logprobs: %v", i, choice.Logprobs)
		t.Logf("choice %d finish-reason: %v", i, choice.FinishReason)
	}
}

func TestCreateEdit(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := c.CreateEdit(ctx, nil, &openai.CreateEditRequest{
		Model:       openai.ModelTextDavinciEdit001,
		Instruction: "Change the word 'test' to 'example'",
		Input:       "This is a test",
	})

	if err != nil {
		t.Fatal(err)
	}

	// Should only have one choice.
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}

	// Should have the output text "This is an example\n"
	if strings.TrimSpace(strings.TrimSuffix(resp.Choices[0].Text, "\n")) != "This is an example" {
		t.Fatalf("expected 'This is an example', got %q", resp.Choices[0].Text)
	}
}

func TestCreateImage(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := c.CreateImage(ctx, nil, &openai.CreateImageRequest{
		Prompt:         "Golang-style gopher mascot wearing an OpenAI t-shirt",
		N:              1,
		Size:           "256x256",
		ResponseFormat: "url",
	})
	if err != nil {
		t.Fatal(err)
	}

	// Should only have one image.
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Data))
	}

	t.Logf("image url: %v", *resp.Data[0].URL)
}
