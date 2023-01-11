package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Client is a client for the OpenAI API.
//
// https://beta.openai.com/docs/api-reference
type Client struct {
	// APIKey is the API key to use for requests.
	APIKey string
}

// NewClient returns a new Client with the given API key.
//
// # Example
//
//	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
func NewClient(apiKey string) *Client {
	return &Client{
		APIKey: apiKey,
	}
}

// CreateCompletionRequest contains information for a "completion" request
// to the OpenAI API. This is the fundamental request type for the API.
//
// https://beta.openai.com/docs/api-reference/completions/create
type CreateCompletionRequest struct {
	// ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
	//
	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-model
	Model string `json:"model"`

	// The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
	//
	// Note that <|endoftext|> is the document separator that the model sees during training, so if a prompt is not specified the model
	// will generate as if from the beginning of a new document.
	//
	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-prompt
	Prompt []string `json:"prompt"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-suffix
	Suffix string `json:"suffix,omitempty"`

	// The maximum number of tokens to generate in the completion.
	//
	// The token count of your prompt plus max_tokens cannot exceed the model's context length. Most models have a context
	// length of 2048 tokens (except for the newest models, which support 4096).
	//
	// Defaults to 16 if not specified.
	//
	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
	MaxTokens int `json:"max_tokens,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-temperature
	//
	// Defaults to 1 if not specified.
	Temperature float64 `json:"temperature,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-top_p
	//
	// Defaults to 1 if not specified.
	TopP float64 `json:"top_p,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-n
	//
	// Defaults to 1 if not specified.
	N int `json:"n,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-stream
	//
	// Defaults to false if not specified.
	Stream bool `json:"stream,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-logprobs
	//
	// Defaults to nil.
	LogProbs *int `json:"logprobs,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-echo
	//
	// Defaults to false if not specified.
	Echo bool `json:"echo,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-stop
	Stop []string `json:"stop,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-presence_penalty
	//
	// Defaults to 0 if not specified.
	PresencePenalty int `json:"presence_penalty,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-frequency_penalty
	//
	// Defaults to 0 if not specified.
	FrequencyPenalty int `json:"frequency_penalty,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-best_of
	//
	// Defaults to 1 if not specified.
	//
	// WARNING: Because this parameter generates many completions, it can quickly consume your token quota.
	//          Use carefully and ensure that you have reasonable settings for max_tokens and stop.
	BestOf int `json:"best_of,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-logit_bias
	//
	// Defaults to nil.
	LogitBias map[string]float64 `json:"logit_bias,omitempty"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-user
	//
	// Defaults to nil.
	User string `json:"user,omitempty"`
}

// CompletionResponse is the response from a "completion" request to the OpenAI API.
//
// https://beta.openai.com/docs/api-reference/completions/create
type CompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Text         string      `json:"text"`
		Index        int         `json:"index"`
		Logprobs     interface{} `json:"logprobs"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// CreateCompletion performs a "completion" request using the OpenAI API.
// This is the fundamental request for the API.
//
// # Example
//
//	 resp, _ := client.CreateCompletion(ctx, nil, &openai.CreateCompletionRequest{
//		Model: openai.ModelDavinci,
//		Prompt: []string{"Once upon a time"},
//		MaxTokens: 16,
//	 })
//
// https://beta.openai.com/docs/api-reference/completions/create
func (c *Client) CreateCompletion(ctx context.Context, h *http.Client, req *CreateCompletionRequest) (*CompletionResponse, error) {
	if h == nil {
		h = http.DefaultClient
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/completions", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")

	resp, err := h.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	cResp := &CompletionResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://beta.openai.com/docs/api-reference/models/list
type Models struct {
	Object string `json:"object"`
	Data   []struct {
		ID         string `json:"id"`
		Object     string `json:"object"`
		Created    int    `json:"created"`
		OwnedBy    string `json:"owned_by"`
		Permission []struct {
			ID                 string      `json:"id"`
			Object             string      `json:"object"`
			Created            int         `json:"created"`
			AllowCreateEngine  bool        `json:"allow_create_engine"`
			AllowSampling      bool        `json:"allow_sampling"`
			AllowLogprobs      bool        `json:"allow_logprobs"`
			AllowSearchIndices bool        `json:"allow_search_indices"`
			AllowView          bool        `json:"allow_view"`
			AllowFineTuning    bool        `json:"allow_fine_tuning"`
			Organization       string      `json:"organization"`
			Group              interface{} `json:"group"`
			IsBlocking         bool        `json:"is_blocking"`
		} `json:"permission"`
		Root   string      `json:"root"`
		Parent interface{} `json:"parent"`
	} `json:"data"`
}

// ListModels list model identifiers that can be used with the OpenAI API.
//
// # Example
//
//	resp, _ := client.ListModels(ctx, nil)
//
//	for _, model := range resp.Data {
//	   fmt.Println(model.ID)
//	}
//
// https://beta.openai.com/docs/api-reference/models/list
func (c *Client) ListModels(ctx context.Context, h *http.Client) (*Models, error) {
	if h == nil {
		h = http.DefaultClient
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/models", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")

	resp, err := h.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	cResp := &Models{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// CreateEditRequest is the request for a "edit" request to the OpenAI API.
//
// https://beta.openai.com/docs/api-reference/edits/create
type CreateEditRequest struct {
	// https://beta.openai.com/docs/api-reference/edits/create#edits/create-model
	//
	// Required.
	Model string `json:"model"`

	// https://beta.openai.com/docs/api-reference/edits/create#edits/create-instruction
	//
	// Required.
	Instruction string `json:"instruction"`

	// https://beta.openai.com/docs/api-reference/edits/create#edits/create-input
	Input string `json:"input"`

	// https://beta.openai.com/docs/api-reference/edits/create#edits/create-n
	N int `json:"n,omitempty"`

	// https://beta.openai.com/docs/api-reference/edits/create#edits/create-temperature
	Temperature float64 `json:"temperature,omitempty"`

	// https://beta.openai.com/docs/api-reference/edits/create#edits/create-top-p
	TopP float64 `json:"top_p,omitempty"`
}

// https://beta.openai.com/docs/api-reference/edits/create
type CreateEditResponse struct {
	Object  string `json:"object"`
	Created int    `json:"created"`
	Choices []struct {
		Text  string `json:"text"`
		Index int    `json:"index"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// CreateEdit performs a "edit" request using the OpenAI API.
//
// # Example
//
//	resp, _ := client.CreateEdit(ctx, nil, &CreateEditRequest{
//		Model:       openai.ModelTextDavinciEdit001,
//		Instruction: "Change the word 'test' to 'example'",
//		Input:       "This is a test",
//	})
//
// https://beta.openai.com/docs/api-reference/edits/create
func (c *Client) CreateEdit(ctx context.Context, h *http.Client, req *CreateEditRequest) (*CreateEditResponse, error) {
	if h == nil {
		h = http.DefaultClient
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/edits", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(b)))

	resp, err := h.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &CreateEditResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://beta.openai.com/docs/api-reference/images/create
type CreateImageRequest struct {
	// https://beta.openai.com/docs/api-reference/images/create#images/create-prompt
	//
	// Required. Max of 1,000 characters.
	Prompt string `json:"prompt"`

	// https://beta.openai.com/docs/api-reference/completions/create#completions/create-n
	//
	// Number of images to generate. Defaults to 1 if not specified. Most be between 1 and 10.
	N int `json:"n,omitempty"`

	// https://beta.openai.com/docs/api-reference/images/create#images/create-size
	//
	// Size of the image to generate. Must be one of 256x256, 512x512, or 1024x1024.
	Size string `json:"size,omitempty"`

	// https://beta.openai.com/docs/api-reference/images/create#images/create-response_format
	//
	// Defaults to "url". The format in which the generated images are returned. Must be one of "url" or "b64_json".
	ResponseFormat string `json:"response_format,omitempty"`

	// https://beta.openai.com/docs/api-reference/images/create#images/create-user
	User string `json:"user,omitempty"`
}

// CreateImageResponse ...
type CreateImageResponse struct {
	Created int `json:"created"`
	Data    []struct {
		// One of the following: "url" or "b64_json"
		URL     *string `json:"url"`
		B64JSON *string `json:"b64_json"`
	} `json:"data"`
}

// CreateImage performs a "image" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.CreateImage(ctx, nil, &openai.CreateImageRequest{
//		Prompt:         "Golang-style gopher mascot wearing an OpenAI t-shirt",
//		N:              1,
//		Size:           "256x256",
//		ResponseFormat: "url",
//	})
//
// https://beta.openai.com/docs/api-reference/images/create
func (c *Client) CreateImage(ctx context.Context, h *http.Client, req *CreateImageRequest) (*CreateImageResponse, error) {
	if h == nil {
		h = http.DefaultClient
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/images/generations", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(b)))

	resp, err := h.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &CreateImageResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil

}

// TODO:
// - https://beta.openai.com/docs/api-reference/images/create-edit
// - https://beta.openai.com/docs/api-reference/images/create-variation
// - https://beta.openai.com/docs/api-reference/embeddings
// - https://beta.openai.com/docs/api-reference/files
// - https://beta.openai.com/docs/api-reference/fine-tunes
// - https://beta.openai.com/docs/api-reference/moderations
