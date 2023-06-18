package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
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

func TestCreateModeration(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	resp, err := c.CreateModeration(ctx, &openai.CreateModerationRequest{
		Input: "I want to kill them.",
	})

	if err != nil {
		t.Fatal(err)
	}

	t.Logf("moderation: %#+v", resp)
}

func TestCreateChat(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	userMessage := "Hello!"

	resp, err := c.CreateChat(ctx, &openai.CreateChatRequest{
		Model: openai.ModelGPT35Turbo,
		Messages: []openai.ChatMessage{
			{
				Role:    "user",
				Content: userMessage,
			},
		},
	})

	if err != nil {
		t.Fatal(err)
	}

	t.Logf("user: %q", userMessage)

	t.Logf("bot: %q", strings.TrimSpace(resp.Choices[0].Message.Content))
}

func TestMessageJSONUnmarshal(t *testing.T) {
	data := `
	{
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "getCurrentLocation",
          "arguments": "{\n  \"location\": \"Ann Arbor, MI\",\n  \"unit\": \"celsius\"\n}"
        }
    }
	`

	var msg openai.ChatMessage

	err := json.Unmarshal([]byte(data), &msg)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("msg: %#+v", msg.FunctionCall.Arguments)

	args := msg.FunctionCall.Arguments

	locationArg, err := openai.GetTypedFunctionCallArgumentValue[string]("location", args)
	if err != nil {
		t.Fatal(err)
	}

	unitArg, err := openai.GetTypedFunctionCallArgumentValue[string]("unit", args)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("location: %[1]T(%[1]q)", locationArg)
	t.Logf("unit: %[1]T(%[1]q)", unitArg)
}

func TestCreateChat_FunctionCall(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	// https://platform.openai.com/docs/guides/gpt/function-calling
	getCurrentWeather := func(location string, unit string) string {
		return fmt.Sprintf("The current temperature in %s is 72 degrees %s.", location, unit)
	}

	resp, err := c.CreateChat(ctx, &openai.CreateChatRequest{
		Model: openai.ModelGPT35Turbo0613,
		Messages: []openai.ChatMessage{
			{
				Role:    "user",
				Content: "What's the weather like in Ann Arbor?",
			},
		},
		FunctionCall: openai.FunctionCallAuto,
		Functions: []*openai.Function{
			{
				Name:        "getCurrentWeather",
				Description: "Gets the current weather in a location from the given location and unit.",
				Parameters: &openai.JSONSchema{
					Type: "object",
					Properties: map[string]*openai.JSONSchema{
						"location": {
							Type:        "string",
							Description: "The city and state, e.g. San Francisco, CA",
						},
						"unit": {
							Type: "string",
							Enum: []string{"fahrenheit", "celsius"},
						},
					},
					Required: []string{"location", "unit"},
				},
			},
		},
	})

	if err != nil {
		t.Fatal(err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "getCurrentWeather" {
		t.Fatalf("expected function name to be %q, got %q", "getCurrentWeather", resp.Choices[0].Message.FunctionCall.Name)
	}

	args := resp.Choices[0].Message.FunctionCall.Arguments

	locationArg, err := openai.GetTypedFunctionCallArgumentValue[string]("location", args)
	if err != nil {
		t.Fatal(err)
	}

	unitArg, err := openai.GetTypedFunctionCallArgumentValue[string]("unit", args)
	if err != nil {
		t.Fatal(err)
	}

	currentWeather := getCurrentWeather(locationArg, unitArg)

	t.Logf("currentWeather: %q", currentWeather)
}

func TestCreateAudioTranscription(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	fh, err := os.Open("testdata/hello-world.m4a")
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()

	resp, err := c.CreateAudioTranscription(ctx, &openai.CreateAudioTranscriptionRequest{
		Model: openai.ModelWhisper1,
		File:  fh,
	})

	if err != nil {
		t.Fatal(err)
	}

	if resp.Text() != "Hello world, from an audio file." {
		t.Fatalf("expected 'Hello world, from an audio file.', got %q", resp.Text())
	}
}

func ExampleClient_CreateCompletion() {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := context.Background()

	resp, err := c.CreateCompletion(ctx, &openai.CreateCompletionRequest{
		Model:     openai.ModelDavinci,
		Prompt:    []string{"The cow jumped over the"},
		MaxTokens: 1,
		N:         1,
	})

	if err != nil {
		panic(err)
	}

	fmt.Println(resp.Choices[0].Text)
	// Output: moon
}

func ExampleClient_CreateEdit() {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := context.Background()

	resp, err := c.CreateEdit(ctx, &openai.CreateEditRequest{
		Model:       openai.ModelTextDavinciEdit001,
		Instruction: "ONLY change the word 'test' to 'example', with no other changes",
		Input:       "This is a test",
	})

	if err != nil {
		panic(err)
	}

	// Get the words from the response.
	words := strings.Split(resp.Choices[0].Text, " ")

	fmt.Println(words[len(words)-1])
	// Output: example
}

func ExampleClient_CreateChat() {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := context.Background()

	messages := []openai.ChatMessage{
		{
			Role:    openai.ChatRoleSystem,
			Content: "You are a helpful assistant familiar with children's stories, and answer in only single words.",
		},
		{
			Role:    "user",
			Content: "Clifford is a big dog, but what color is he?",
		},
	}

	resp, err := c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:    openai.ModelGPT35Turbo,
		Messages: messages,
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(strings.ToLower(strings.TrimRight(strings.TrimSpace(resp.Choices[0].Message.Content), ".")))
	// Output: red
}
