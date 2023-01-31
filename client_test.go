package openai_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/picatz/openai"
)

func testCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestCreateCompletion(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	resp, err := c.CreateCompletion(ctx, &openai.CreateCompletionRequest{
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

	ctx := testCtx(t)

	resp, err := c.CreateEdit(ctx, &openai.CreateEditRequest{
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

	ctx := testCtx(t)

	resp, err := c.CreateImage(ctx, &openai.CreateImageRequest{
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

func TestCreateEmbedding(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	resp, err := c.CreateEmbedding(ctx, &openai.CreateEmbeddingRequest{
		Model: openai.ModelTextEmbeddingAda002,
		Input: "The food was delicious and the waiter...",
	})

	if err != nil {
		t.Fatal(err)
	}

	t.Logf("embedding: %#+v", resp)
}
