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

func TestCreateChat_Stream(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	resp, err := c.CreateChat(ctx, &openai.CreateChatRequest{
		Model: openai.ModelGPT35Turbo,
		Messages: []openai.ChatMessage{
			{
				Role:    "user",
				Content: "Hello!",
			},
		},
		Stream: true,
	})

	if err != nil {
		t.Fatal(err)
	}

	defer resp.Stream.Close()

	b := strings.Builder{}

	err = resp.ReadStream(ctx, func(c *openai.ChatMessageStreamChunk) error {
		if c.ContentDelta() {
			contentChunk, err := c.FirstChoice()
			if err != nil {
				return err
			}
			b.WriteString(contentChunk)
		}
		return nil
	})

	t.Logf("bot: %q", strings.TrimSpace(b.String()))

	if err != nil {
		t.Fatal(err)
	}
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

	locationArg, err := openai.FunctionCallArgumentValue[string]("location", args)
	if err != nil {
		t.Fatal(err)
	}

	unitArg, err := openai.FunctionCallArgumentValue[string]("unit", args)
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

	locationArg, err := openai.FunctionCallArgumentValue[string]("location", args)
	if err != nil {
		t.Fatal(err)
	}

	unitArg, err := openai.FunctionCallArgumentValue[string]("unit", args)
	if err != nil {
		t.Fatal(err)
	}

	currentWeather := getCurrentWeather(locationArg, unitArg)

	t.Logf("currentWeather: %q", currentWeather)
}

func TestCreateChat_FunctionCall_AssistantAgent(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	// fn is a generic way to define a function that can be called by the model
	type fn struct {
		description string
		parameters  *openai.JSONSchema
		call        func(...any) string
	}

	// fnMap are a bunch of named functions that can be called by the model
	fnMap := map[string]fn{
		"getCurrentWeather": {
			description: "Gets the current weather in a location from the given location and unit.",
			parameters: &openai.JSONSchema{
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
			call: func(args ...any) string {
				locationArg := args[0].(string)
				unitArg := args[1].(string)
				return fmt.Sprintf("The current temperature in %s is 72 degrees %s.", locationArg, unitArg)
			},
		},
		"remindMe": {
			description: "Reminds the user of something at a given time.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"reminder": {
						Type:        "string",
						Description: "The reminder to set.",
					},
					"time": {
						Type:        "string",
						Description: "The time to set the reminder for.",
					},
				},
				Required: []string{"reminder", "time"},
			},
			call: func(args ...any) string {
				reminderArg := args[0].(string)
				timeArg := args[1].(string)
				return fmt.Sprintf("Reminder set for %s: %s", timeArg, reminderArg)
			},
		},
		"timer": {
			description: "Sets a timer for a given amount of time.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"duration": {
						Type:        "string",
						Description: "The duration of the timer.",
					},
				},
				Required: []string{"duration"},
			},
			call: func(args ...any) string {
				durationArg := args[0].(string)
				return fmt.Sprintf("Timer set for %s", durationArg)
			},
		},
		"setAlarm": {
			description: "Sets an alarm for a given time.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"time": {
						Type:        "string",
						Description: "The time to set the alarm for.",
					},
				},
				Required: []string{"time"},
			},
			call: func(args ...any) string {
				timeArg := args[0].(string)
				return fmt.Sprintf("Alarm set for %s", timeArg)
			},
		},
		"scheduleMeeting": {
			description: "Schedules a meeting for a given time.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"time": {
						Type:        "string",
						Description: "The time to schedule the meeting for.",
					},
					"people": {
						Type: "array",
						Items: &openai.JSONSchema{
							Type:        "string",
							Description: "The person to invite to the meeting.",
						},
						Description: "The people to invite to the meeting.",
					},
					"message": {
						Type:        "string",
						Description: "The message to send to the people invited to the meeting.",
					},
				},
				Required: []string{"time", "people", "message"},
			},
			call: func(args ...any) string {
				timeArg := args[0].(string)
				peopleArg := args[1].([]string)
				messageArg := args[2].(string)

				people := strings.Join(peopleArg, ", ")

				return fmt.Sprintf("Meeting scheduled for %s with %s. Message: %s", timeArg, people, messageArg)
			},
		},
	}

	// seed messages for the test chat
	openaiMessages := []openai.ChatMessage{
		{
			Role:    "system",
			Content: "You are a helpful assitant. It's currently 9:00 AM, and the user is in Ann Arbor, Michigan, USA.",
		},
		{
			Role:    "user",
			Content: "What's the weather going to be like today?",
		},
	}

	// functions that the model can call, in the right format for the API
	openaiFunctions := func() []*openai.Function {
		fns := []*openai.Function{}
		for name, fn := range fnMap {
			fns = append(fns, &openai.Function{
				Name:        name,
				Description: fn.description,
				Parameters:  fn.parameters,
			})
		}
		return fns
	}()

	resp, err := c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:        openai.ModelGPT35Turbo0613,
		Messages:     openaiMessages,
		FunctionCall: openai.FunctionCallAuto,
		Functions:    openaiFunctions,
	})

	if err != nil {
		t.Fatal(err)
	}

	// Add the response to the messages
	openaiMessages = append(openaiMessages, resp.Choices[0].Message)

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	t.Logf("resp: %#+v", resp.Choices[0].Message)

	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "getCurrentWeather" {
		t.Fatalf("expected function name to be %q, got %q", "getCurrentWeather", resp.Choices[0].Message.FunctionCall.Name)
	}

	fnName := resp.Choices[0].Message.FunctionCall.Name

	args := resp.Choices[0].Message.FunctionCall.Arguments

	locationArg, err := openai.FunctionCallArgumentValue[string]("location", args)
	if err != nil {
		t.Fatal(err)
	}

	unitArg, err := openai.FunctionCallArgumentValue[string]("unit", args)
	if err != nil {
		t.Fatal(err)
	}

	currentWeather := fnMap[fnName].call(locationArg, unitArg)

	t.Logf("%s", currentWeather)

	// Add the current weather to the messages
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "assistant",
		Content: currentWeather,
	})

	// Now ask the assistant to remind us to do something
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "user",
		Content: "Remind me to take out the trash at 8:00 PM",
	})

	resp, err = c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:        openai.ModelGPT35Turbo0613,
		Messages:     openaiMessages,
		FunctionCall: openai.FunctionCallAuto,
		Functions:    openaiFunctions,
	})

	if err != nil {
		t.Fatal(err)
	}

	// Add the response to the messages
	openaiMessages = append(openaiMessages, resp.Choices[0].Message)

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	t.Logf("resp: %#+v", resp.Choices[0].Message)

	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "remindMe" {
		t.Fatalf("expected function name to be %q, got %q", "remindMe", resp.Choices[0].Message.FunctionCall.Name)
	}

	fnName = resp.Choices[0].Message.FunctionCall.Name

	args = resp.Choices[0].Message.FunctionCall.Arguments

	reminderArg, err := openai.FunctionCallArgumentValue[string]("reminder", args)
	if err != nil {
		t.Fatal(err)
	}

	timeArg, err := openai.FunctionCallArgumentValue[string]("time", args)
	if err != nil {
		t.Fatal(err)
	}

	reminder := fnMap[fnName].call(reminderArg, timeArg)

	t.Logf("%s", reminder)

	// Add the reminder to the messages
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "assistant",
		Content: reminder,
	})

	// Now ask the assistant to set a timer
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "user",
		Content: "Set a timer for 5 minutes",
	})

	resp, err = c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:        openai.ModelGPT35Turbo0613,
		Messages:     openaiMessages,
		FunctionCall: openai.FunctionCallAuto,
		Functions:    openaiFunctions,
	})

	if err != nil {
		t.Fatal(err)
	}

	// Add the response to the messages
	openaiMessages = append(openaiMessages, resp.Choices[0].Message)

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	t.Logf("resp: %#+v", resp.Choices[0].Message)

	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "timer" {
		t.Fatalf("expected function name to be %q, got %q", "timer", resp.Choices[0].Message.FunctionCall.Name)
	}

	fnName = resp.Choices[0].Message.FunctionCall.Name

	args = resp.Choices[0].Message.FunctionCall.Arguments

	durationArg, err := openai.FunctionCallArgumentValue[string]("duration", args)
	if err != nil {
		t.Fatal(err)
	}

	timer := fnMap[fnName].call(durationArg)

	t.Logf("%s", timer)

	// Add the timer to the messages
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "assistant",
		Content: timer,
	})

	// Now ask the assistant to set an alarm
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "user",
		Content: "Set an alarm for 8:00 PM",
	})

	resp, err = c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:        openai.ModelGPT35Turbo0613,
		Messages:     openaiMessages,
		FunctionCall: openai.FunctionCallAuto,
		Functions:    openaiFunctions,
	})

	if err != nil {
		t.Fatal(err)
	}

	// Add the response to the messages
	openaiMessages = append(openaiMessages, resp.Choices[0].Message)

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	t.Logf("resp: %#+v", resp.Choices[0].Message)

	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "setAlarm" {
		t.Fatalf("expected function name to be %q, got %q", "setAlarm", resp.Choices[0].Message.FunctionCall.Name)
	}

	fnName = resp.Choices[0].Message.FunctionCall.Name

	args = resp.Choices[0].Message.FunctionCall.Arguments

	timeArg, err = openai.FunctionCallArgumentValue[string]("time", args)
	if err != nil {
		t.Fatal(err)
	}

	alarm := fnMap[fnName].call(timeArg)

	t.Logf("%s", alarm)

	// schedule a meeting
	openaiMessages = append(openaiMessages, openai.ChatMessage{
		Role:    "user",
		Content: "Schedule a meeting for 3:00 PM with John, Jane, and Joe to talk about our favorite books.",
	})

	resp, err = c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:        openai.ModelGPT35Turbo0613,
		Messages:     openaiMessages,
		FunctionCall: openai.FunctionCallAuto,
		Functions:    openaiFunctions,
	})
	if err != nil {
		t.Fatal(err)
	}

	// check if the meeting was scheduled
	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "scheduleMeeting" {
		t.Fatalf("expected function name to be %q, got %q", "scheduleMeeting", resp.Choices[0].Message.FunctionCall.Name)
	}

	fnName = resp.Choices[0].Message.FunctionCall.Name

	args = resp.Choices[0].Message.FunctionCall.Arguments

	timeArg, err = openai.FunctionCallArgumentValue[string]("time", args)
	if err != nil {
		t.Fatal(err)
	}

	peopleAnyArg, err := openai.FunctionCallArgumentValue[[]any]("people", args)
	if err != nil {
		t.Fatal(err)
	}

	peopleArg := make([]string, len(peopleAnyArg))
	for i, v := range peopleAnyArg {
		peopleArg[i] = v.(string)
	}

	messageArg, err := openai.FunctionCallArgumentValue[string]("message", args)
	if err != nil {
		t.Fatal(err)
	}

	meeting := fnMap[fnName].call(timeArg, peopleArg, messageArg)

	t.Logf("%s", meeting)
}

func TestCreateChat_FunctionCall_LinuxAssistantAgent(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := testCtx(t)

	// fn is a generic way to define a function that can be called by the model
	type fn struct {
		description string
		parameters  *openai.JSONSchema
		call        func(...any) string
	}

	// fnMap are a bunch of named functions that can be called by the model
	fnMap := map[string]fn{
		"run": {
			description: "Runs a command on the Linux machine.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"command": {
						Type:        "string",
						Description: "The command to run.",
					},
				},
				Required: []string{"command"},
			},
			call: func(args ...any) string {
				commandArg := args[0].(string)
				return fmt.Sprintf("Running command: %s", commandArg)
			},
		},
		"connect": {
			description: "Connects to a Linux machine using SSH.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"host": {
						Type:        "string",
						Description: "The host to connect to.",
					},
					"username": {
						Type:        "string",
						Description: "The username to use when connecting.",
					},
				},
				Required: []string{"host"},
			},
			call: func(args ...any) string {
				hostArg := args[0].(string)

				if usernameArg, ok := args[1].(string); ok {
					return fmt.Sprintf("Connected to %q as %q", hostArg, usernameArg)
				}

				return fmt.Sprintf("Connected to %q as %q", hostArg, "root")
			},
		},
		"disconnect": {
			description: "Disconnects from a Linux machine.",
			parameters: &openai.JSONSchema{
				Type: "object",
				Properties: map[string]*openai.JSONSchema{
					"message": {
						Type:        "string",
						Description: "The short, fun (fortune style) message to display when disconnecting.",
					},
				},
				Required: []string{"message"},
			},
			call: func(args ...any) string {
				return "Disconnected"
			},
		},
	}

	// seed messages for the test chat
	openaiMessages := []openai.ChatMessage{
		{
			Role:    "system",
			Content: "You are a helpful assistant to use a Linux machine with natural language.",
		},
		{
			Role:    "user",
			Content: "What system am I using?",
		},
	}

	// functions that the model can call, in the right format for the API
	openaiFunctions := func() []*openai.Function {
		fns := []*openai.Function{}
		for name, fn := range fnMap {
			fns = append(fns, &openai.Function{
				Name:        name,
				Description: fn.description,
				Parameters:  fn.parameters,
			})
		}
		return fns
	}()

	resp, err := c.CreateChat(ctx, &openai.CreateChatRequest{
		Model:        openai.ModelGPT35Turbo0613,
		Messages:     openaiMessages,
		FunctionCall: openai.FunctionCallAuto,
		Functions:    openaiFunctions,
	})

	if err != nil {
		t.Fatal(err)
	}

	// Add the response to the messages
	// openaiMessages = append(openaiMessages, resp.Choices[0].Message)

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	if resp.Choices[0].Message.FunctionCall == nil {
		t.Fatal("expected function to be non-nil")
	}

	if resp.Choices[0].Message.FunctionCall.Name != "run" {
		t.Fatalf("expected function name to be %q, got %q", "run", resp.Choices[0].Message.FunctionCall.Name)
	}

	fnName := resp.Choices[0].Message.FunctionCall.Name

	args := resp.Choices[0].Message.FunctionCall.Arguments

	commandArg, err := openai.FunctionCallArgumentValue[string]("command", args)
	if err != nil {
		t.Fatal(err)
	}

	command := fnMap[fnName].call(commandArg)

	// should contain the "uname" command
	if !strings.Contains(command, "uname") {
		t.Fatalf("expected command to contain %q, got %q", "uname", command)
	}

	t.Logf("%s", command)
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

// func TestAssistant_list_delete(t *testing.T) {
// 	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
//
// 	ctx := context.Background()
//
// 	resp, err := c.ListAssistants(ctx, &openai.ListAssistantsRequest{
// 		Limit: 100,
// 	})
// 	if err != nil {
// 		t.Fatal(err)
// 	}
//
// 	for _, assistant := range resp.Data {
// 		t.Logf("assistant: %#+v", assistant)
//
// 		err := c.DeleteAssistant(ctx, &openai.DeleteAssistantRequest{
// 			ID: assistant.ID,
// 		})
// 		if err != nil {
// 			t.Fatal(err)
// 		}
// 	}
//
// 	t.Logf("assistants: %#+v", resp)
// }

func TestAssistant_beta(t *testing.T) {
	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	ctx := context.Background()

	var assistant *openai.Assistant

	t.Run("create", func(t *testing.T) {
		var err error
		assistant, err = c.CreateAssistant(ctx, &openai.CreateAssistantRequest{
			Name:         "Test Assistant",
			Instructions: "You are a helpful assistant.",
			Model:        openai.ModelGPT41106Previw,
			Tools: []map[string]any{
				{
					"type": "code_interpreter",
				},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("list", func(t *testing.T) {
		resp, err := c.ListAssistants(ctx, &openai.ListAssistantsRequest{
			Limit: 1,
		})
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Data) != 1 {
			t.Fatalf("expected 1 assistant, got %d", len(resp.Data))
		}
	})

	t.Run("get", func(t *testing.T) {
		resp, err := c.GetAssistant(ctx, &openai.GetAssistantRequest{
			ID: assistant.ID,
		})
		if err != nil {
			t.Fatal(err)
		}

		if resp.ID != assistant.ID {
			t.Fatalf("expected assistant ID %q, got %q", assistant.ID, resp.ID)
		}
	})

	t.Run("update", func(t *testing.T) {
		resp, err := c.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
			ID:           assistant.ID,
			Instructions: "Always respond with 'Hello, world!'",
			Metadata: map[string]any{
				"foo": "bar",
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if resp.ID != assistant.ID {
			t.Fatalf("expected assistant ID %q, got %q", assistant.ID, resp.ID)
		}

		if resp.Metadata["foo"] != "bar" {
			t.Fatal("expected metadata to be updated")
		}

		if resp.Instructions != "Always respond with 'Hello, world!'" {
			t.Fatal("expected instructions to be updated")
		}
	})

	t.Run("delete", func(t *testing.T) {
		err := c.DeleteAssistant(ctx, &openai.DeleteAssistantRequest{
			ID: assistant.ID,
		})
		if err != nil {
			t.Fatal(err)
		}
	})
}
