package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
)

// Client is a client for the OpenAI API.
//
// https://beta.openai.com/docs/api-reference
type Client struct {
	// APIKey is the API key to use for requests.
	APIKey string

	// HTTPClient is the HTTP client to use for requests.
	HTTPClient *http.Client

	// Organization is the organization to use for requests.
	Organization string
}

// ClientOption is a function that configures a Client.
type ClientOption func(*Client)

// WithHTTPClient is a ClientOption that sets the HTTP client to use for requests.
//
// If the client is nil, then http.DefaultClient is used
func WithHTTPClient(c *http.Client) ClientOption {
	return func(client *Client) {
		if c == nil {
			c = http.DefaultClient
		}
		client.HTTPClient = c
	}
}

// WithOrganization is a ClientOption that sets the organization to use for requests.
//
// https://beta.openai.com/docs/api-reference/authentication
func WithOrganization(org string) ClientOption {
	return func(client *Client) {
		client.Organization = org
	}
}

// NewClient returns a new Client with the given API key.
//
// # Example
//
//	c := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
func NewClient(apiKey string, opts ...ClientOption) *Client {
	c := &Client{
		APIKey:     apiKey,
		HTTPClient: http.DefaultClient,
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
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

// CreateCompletionResponse is the response from a "completion" request to the OpenAI API.
//
// https://beta.openai.com/docs/api-reference/completions/create
type CreateCompletionResponse struct {
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
//	 resp, _ := client.CreateCompletion(ctx, &openai.CreateCompletionRequest{
//		Model: openai.ModelDavinci,
//		Prompt: []string{"Once upon a time"},
//		MaxTokens: 16,
//	 })
//
// https://beta.openai.com/docs/api-reference/completions/create
func (c *Client) CreateCompletion(ctx context.Context, req *CreateCompletionRequest) (*CreateCompletionResponse, error) {
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

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	cResp := &CreateCompletionResponse{}
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
//	resp, _ := client.ListModels(ctx)
//
//	for _, model := range resp.Data {
//	   fmt.Println(model.ID)
//	}
//
// https://beta.openai.com/docs/api-reference/models/list
func (c *Client) ListModels(ctx context.Context) (*Models, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/models", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
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
//	resp, _ := client.CreateEdit(ctx, &CreateEditRequest{
//		Model:       openai.ModelTextDavinciEdit001,
//		Instruction: "Change the word 'test' to 'example'",
//		Input:       "This is a test",
//	})
//
// https://beta.openai.com/docs/api-reference/edits/create
func (c *Client) CreateEdit(ctx context.Context, req *CreateEditRequest) (*CreateEditResponse, error) {
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

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
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
//	resp, _ := c.CreateImage(ctx, &openai.CreateImageRequest{
//		Prompt:         "Golang-style gopher mascot wearing an OpenAI t-shirt",
//		N:              1,
//		Size:           "256x256",
//		ResponseFormat: "url",
//	})
//
// https://beta.openai.com/docs/api-reference/images/create
func (c *Client) CreateImage(ctx context.Context, req *CreateImageRequest) (*CreateImageResponse, error) {
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

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
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

// https://platform.openai.com/docs/api-reference/embeddings
type CreateEmbeddingRequest struct {
	// https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-model
	//
	// Required. The text to embed.
	Model string `json:"model"`

	// https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-input
	//
	// Required. The text to embed.
	Input string `json:"input"`

	// https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-user
	User string `json:"user,omitempty"`
}

// CreateEmbeddingResponse ...
//
// https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
type CreateEmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// CreateEmbedding performs a "embedding" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.CreateEmbedding(ctx, &openai.CreateEmbeddingRequest{
//		Model: openai.ModelTextEmbeddingAda002,
//		Input: "The food was delicious and the waiter...",
//	})
//
// https://platform.openai.com/docs/api-reference/embeddings
func (c *Client) CreateEmbedding(ctx context.Context, req *CreateEmbeddingRequest) (*CreateEmbeddingResponse, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/embeddings", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(b)))

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &CreateEmbeddingResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://platform.openai.com/docs/api-reference/moderations/create
type CreateModerationRequest struct {
	// https://platform.openai.com/docs/api-reference/moderations/create#moderations/create-model
	//
	// Optional. The model to use for moderation. Defaults to "text-moderation-latest".
	Model string `json:"model"`

	// https://platform.openai.com/docs/api-reference/moderations/create#moderations/create-input
	//
	// Required. The text to moderate.
	Input string `json:"input"`
}

// CreateModerationResponse ...
//
// https://platform.openai.com/docs/guides/moderations/what-are-moderations
type CreateModerationResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Results []struct {
		Categories struct {
			Hate            bool `json:"hate"`
			HateThreatening bool `json:"hate/threatening"`
			SelfHarm        bool `json:"self-harm"`
			Sexual          bool `json:"sexual"`
			SexualMinors    bool `json:"sexual/minors"`
			Violence        bool `json:"violence"`
			ViolenceGraphic bool `json:"violence/graphic"`
		} `json:"categories"`
		CategoryScores struct {
			Hate            float64 `json:"hate"`
			HateThreatening float64 `json:"hate/threatening"`
			SelfHarm        float64 `json:"self-harm"`
			Sexual          float64 `json:"sexual"`
			SexualMinors    float64 `json:"sexual/minors"`
			Violence        float64 `json:"violence"`
			ViolenceGraphic float64 `json:"violence/graphic"`
		} `json:"category_scores"`
		Flagged bool `json:"flagged"`
	} `json:"results"`
}

// CreateModeration performs a "moderation" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.CreateModeration(ctx, &openai.CreateModerationRequest{
//		Input: "I want to kill them.",
//	})
//
// https://platform.openai.com/docs/api-reference/moderations
func (c *Client) CreateModeration(ctx context.Context, req *CreateModerationRequest) (*CreateModerationResponse, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/moderations", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(b)))

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &CreateModerationResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://platform.openai.com/docs/api-reference/files/list
type ListFilesRequest struct {
	// There are currently no parameters for this endpoint?
}

// ListFilesResponse ...
//
// https://platform.openai.com/docs/api-reference/files/list
type ListFilesResponse struct {
	Data []struct {
		ID        string `json:"id"`
		Object    string `json:"object"`
		Bytes     int    `json:"bytes"`
		CreatedAt int    `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	} `json:"data"`
	Object string `json:"object"`
}

// ListFiles performs a "list files" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.ListFiles(ctx, &openai.ListFilesRequest{})
//
// https://platform.openai.com/docs/api-reference/files
func (c *Client) ListFiles(ctx context.Context, req *ListFilesRequest) (*ListFilesResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/files", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &ListFilesResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://platform.openai.com/docs/api-reference/files/upload
type UploadFileRequest struct {
	// Name of the JSON Lines file to be uploaded.
	//
	// If the purpose is set to "fine-tune", each line is a JSON
	// record with "prompt" and "completion" fields representing
	// your training examples.
	//
	// Required.
	Name string `json:"name"`

	// Purpose of the uploaded documents.
	//
	// Use "fine-tune" for Fine-tuning. This allows us to validate t
	// the format of the uploaded file.
	//
	// Required.
	Purpose string `json:"purpose"`

	// Body of the file to upload.
	//
	// Required.
	Body io.Reader `json:"file"` // TODO: how to handle this?
}

// UploadFileResponse ...
//
// https://platform.openai.com/docs/api-reference/files/upload
type UploadFileResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int    `json:"bytes"`
	CreatedAt int    `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// UploadFile performs a "upload file" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.UploadFile(ctx, &openai.UploadFileRequest{
//		Name:    "fine-tune.jsonl",
//		Purpose: "fine-tune",
//	})
//
// # CURL
//
//	$ curl "https://api.openai.com/v1/files" \
//	 -H "Authorization: Bearer ..." \
//	 -F purpose="fine-tune" \
//	 -F file='@mydata.jsonl'
//
// https://platform.openai.com/docs/api-reference/files
func (c *Client) UploadFile(ctx context.Context, req *UploadFileRequest) (*UploadFileResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/files", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	r.Header.Set("Content-Type", "multipart/form-data")

	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	fw, err := w.CreateFormFile("file", req.Name)
	if err != nil {
		return nil, err
	}

	_, err = io.Copy(fw, req.Body)
	if err != nil {
		return nil, err
	}

	err = w.WriteField("purpose", req.Purpose)
	if err != nil {
		return nil, err
	}

	err = w.Close()
	if err != nil {
		return nil, err
	}

	r.Body = io.NopCloser(&b)
	r.ContentLength = int64(b.Len())
	r.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &UploadFileResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://platform.openai.com/docs/api-reference/files/delete
type DeleteFileRequest struct {
	// ID of the file to delete.
	//
	// Required.
	ID string `json:"id"`
}

// DeleteFileResponse ...
//
// https://platform.openai.com/docs/api-reference/files/delete
type DeleteFileResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// DeleteFile performs a "delete file" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.DeleteFile(ctx, &openai.DeleteFileRequest{
//		ID: "file-123",
//	})
//
// # CURL
//
//	$ curl "https://api.openai.com/v1/files/file-XjGxS3KTG0uNmNOK362iJua3" \
//		-X DELETE \
//		-H "Authorization: Bearer ..."
//
// https://platform.openai.com/docs/api-reference/files/delete
func (c *Client) DeleteFile(ctx context.Context, req *DeleteFileRequest) (*DeleteFileResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodDelete, "https://api.openai.com/v1/files/"+req.ID, nil)
	if err != nil {
		return nil, err
	}

	r.Header.Set("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &DeleteFileResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://platform.openai.com/docs/api-reference/files/retrieve
type GetFileInfoRequest struct {
	// ID of the file to retrieve.
	//
	// Required.
	ID string `json:"id"`
}

// GetFileInfoResponse ...
//
// https://platform.openai.com/docs/api-reference/files/retrieve
type GetFileInfoResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int    `json:"bytes"`
	CreatedAt int    `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// GetFileInfo performs a "get file info (retrieve)" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.GetFileInfo(ctx, &openai.GetFileRequest{
//		ID: "file-123",
//	})
//
// # CURL
//
//	$ curl "https://api.openai.com/v1/files/file-XjGxS3KTG0uNmNOK362iJua3" \
//		-H "Authorization: Bearer ..."
//
// https://platform.openai.com/docs/api-reference/files/retrieve
func (c *Client) GetFileInfo(ctx context.Context, req *GetFileInfoRequest) (*GetFileInfoResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/files/"+req.ID, nil)
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	cResp := &GetFileInfoResponse{}
	err = json.NewDecoder(resp.Body).Decode(cResp)
	if err != nil {
		return nil, err
	}

	return cResp, nil
}

// https://platform.openai.com/docs/api-reference/files/retrieve-content
type GetFileContentRequest struct {
	// ID of the file to retrieve.
	//
	// Required.
	ID string `json:"id"`
}

// GetFileContentResponse ...
//
// https://platform.openai.com/docs/api-reference/files/retrieve-content
type GetFileContentResponse struct {
	// Body is the file content returned by the OpenAI API.
	//
	// The caller is responsible for closing the body, and should do so as soon as possible.
	Body io.ReadCloser
}

// GetFileContent performs a "get file content (retrieve content)" request using the OpenAI API.
//
// # Example
//
//	resp, _ := c.GetFileContent(ctx, &openai.GetFileContentRequest{
//		ID: "file-123",
//	})
//
// # CURL
//
//	$ curl "https://api.openai.com/v1/files/file-XjGxS3KTG0uNmNOK362iJua3/contents" \
//		-H "Authorization: Bearer ..."
//
// https://platform.openai.com/docs/api-reference/files/retrieve-content
func (c *Client) GetFileContent(ctx context.Context, req *GetFileContentRequest) (*GetFileContentResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/files/"+req.ID+"/contents", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	return &GetFileContentResponse{
		Body: resp.Body,
	}, nil
}

// https://platform.openai.com/docs/api-reference/fine-tunes/create
type CreateFineTuneRequest struct {
	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-training_file
	//
	// Required.
	TrainingFile string `json:"training_file"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-validation_file
	//
	// Optional.
	ValidationFile string `json:"validation_file,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-model
	//
	// Optional. Defaults to "curie".
	Model string `json:"model,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-epochs
	//
	// Optional. Defaults to 4.
	Epochs int `json:"n_epochs,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-batch_size
	//
	// Optional. Defaults to 32.
	BatchSize int `json:"batch_size,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-learning_rate_multiplier
	//
	// Optional. Default depends on the batch size.
	LearningRateMultiplier float64 `json:"learning_rate_multiplier,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-prompt_loss_weight
	//
	// Optional. Defaults to 0.01
	PromptLossWeight float64 `json:"prompt_loss_weight,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-compute_classification_metrics
	//
	// Optional. Defaults to false.
	ComputeClassificationMetrics bool `json:"compute_classification_metrics,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_n_classes
	//
	// Optional, but required for multi-class classification.
	ClassificationNClasses int `json:"classification_n_classes,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_positive_class
	//
	// Optional, but required for binary classification.
	ClassificationPositiveClass string `json:"classification_positive_class,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_betas
	//
	// Optional, only used for binary classification.
	ClassificationBetas []float64 `json:"classification_betas,omitempty"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-suffix
	//
	// A string of up to 40 characters that will be added to your fine-tuned model name.
	//
	// For example, a suffix of "custom-model-name" would produce a model name like
	// `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`.
	//
	// Optional.
	Suffix string `json:"suffix,omitempty"`
}

// CreateFineTuneResponse is the response from a "create fine-tune" request.
//
// https://platform.openai.com/docs/api-reference/fine-tunes/create
type CreateFineTuneResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Model     string `json:"model"`
	CreatedAt int    `json:"created_at"`
	Events    []struct {
		Object    string `json:"object"`
		CreatedAt int    `json:"created_at"`
		Level     string `json:"level"`
		Message   string `json:"message"`
	} `json:"events"`
	FineTunedModel interface{} `json:"fine_tuned_model"`
	Hyperparams    struct {
		BatchSize              int     `json:"batch_size"`
		LearningRateMultiplier float64 `json:"learning_rate_multiplier"`
		NEpochs                int     `json:"n_epochs"`
		PromptLossWeight       float64 `json:"prompt_loss_weight"`
	} `json:"hyperparams"`
	OrganizationID  string        `json:"organization_id"`
	ResultFiles     []interface{} `json:"result_files"`
	Status          string        `json:"status"`
	ValidationFiles []interface{} `json:"validation_files"`
	TrainingFiles   []struct {
		ID        string `json:"id"`
		Object    string `json:"object"`
		Bytes     int    `json:"bytes"`
		CreatedAt int    `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	} `json:"training_files"`
	UpdatedAt int `json:"updated_at"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/create
func (c *Client) CreateFineTune(ctx context.Context, req *CreateFineTuneRequest) (*CreateFineTuneResponse, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/fine-tunes", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res CreateFineTuneResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &res, nil
}

// https://platform.openai.com/docs/api-reference/fine-tunes/list
type ListFineTunesRequest struct {
	// No fields yet.
}

// https://platform.openai.com/docs/api-reference/fine-tunes/list
type ListFineTunesResponse struct {
	Object string `json:"object"`
	Data   []struct {
		ID              string         `json:"id"`
		Object          string         `json:"object"`
		Model           string         `json:"model"`
		CreatedAt       int            `json:"created_at"`
		FineTunedModel  any            `json:"fine_tuned_model"`
		Hyperparams     map[string]any `json:"hyperparams"`
		OrganizationID  string         `json:"organization_id"`
		ResultFiles     []any          `json:"result_files"`
		Status          string         `json:"status"`
		ValidationFiles []any          `json:"validation_files"`
		TrainingFiles   []any          `json:"training_files"`
		UpdatedAt       int            `json:"updated_at"`
	} `json:"data"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/list
func (c *Client) ListFineTunes(ctx context.Context, req *ListFineTunesRequest) (*ListFineTunesResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/fine-tunes", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res ListFineTunesResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &res, nil
}

// https://platform.openai.com/docs/api-reference/fine-tunes/retrieve
type GetFineTuneRequest struct {
	ID string `json:"id"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/retrieve
type GetFineTuneResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Model     string `json:"model"`
	CreatedAt int    `json:"created_at"`
	Events    []struct {
		Object    string `json:"object"`
		CreatedAt int    `json:"created_at"`
		Level     string `json:"level"`
		Message   string `json:"message"`
	} `json:"events"`
	FineTunedModel string `json:"fine_tuned_model"`
	Hyperparams    struct {
		BatchSize              int     `json:"batch_size"`
		LearningRateMultiplier float64 `json:"learning_rate_multiplier"`
		NEpochs                int     `json:"n_epochs"`
		PromptLossWeight       float64 `json:"prompt_loss_weight"`
	} `json:"hyperparams"`
	OrganizationID string `json:"organization_id"`
	ResultFiles    []struct {
		ID        string `json:"id"`
		Object    string `json:"object"`
		Bytes     int    `json:"bytes"`
		CreatedAt int    `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	} `json:"result_files"`
	Status          string `json:"status"`
	ValidationFiles []any  `json:"validation_files"`
	TrainingFiles   []struct {
		ID        string `json:"id"`
		Object    string `json:"object"`
		Bytes     int    `json:"bytes"`
		CreatedAt int    `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	} `json:"training_files"`
	UpdatedAt int `json:"updated_at"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/retrieve
func (c *Client) GetFineTune(ctx context.Context, req *GetFineTuneRequest) (*GetFineTuneResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/fine-tunes/"+req.ID, nil)
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res GetFineTuneResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &res, nil
}

// https://platform.openai.com/docs/api-reference/fine-tunes/cancel
type CancelFineTuneRequest struct {
	ID string `json:"id"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/cancel
type CancelFineTuneResponse struct {
	ID              string `json:"id"`
	Object          string `json:"object"`
	Model           string `json:"model"`
	CreatedAt       int    `json:"created_at"`
	Events          []any  `json:"events"`
	FineTunedModel  any    `json:"fine_tuned_model"`
	Hyperparams     any    `json:"hyperparams"`
	OrganizationID  string `json:"organization_id"`
	ResultFiles     []any  `json:"result_files"`
	Status          string `json:"status"`
	ValidationFiles []any  `json:"validation_files"`
	TrainingFiles   []struct {
		ID        string `json:"id"`
		Object    string `json:"object"`
		Bytes     int    `json:"bytes"`
		CreatedAt int    `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	} `json:"training_files"`
	UpdatedAt int `json:"updated_at"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/cancel
func (c *Client) CancelFineTune(ctx context.Context, req *CancelFineTuneRequest) (*CancelFineTuneResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/fine-tunes/"+req.ID+"/cancel", nil)
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res CancelFineTuneResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &res, nil
}

// https://platform.openai.com/docs/api-reference/fine-tunes/events
type ListFineTuneEventsRequest struct {
	// https://platform.openai.com/docs/api-reference/fine-tunes/events#fine-tunes/events-fine_tune_id
	//
	// Required.
	ID string `json:"id"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/events#fine-tunes/events-stream
	//
	// Optional.
	Stream bool `json:"stream"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/events
type ListFineTuneEventsResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string `json:"object"`
		CreatedAt int    `json:"created_at"`
		Level     string `json:"level"`
		Message   string `json:"message"`
	} `json:"data"`

	// https://platform.openai.com/docs/api-reference/fine-tunes/events#fine-tunes/events-stream
	//
	// Only present if stream=true. Up to the caller to close the stream, e.g.: defer res.Stream.Close()
	Stream io.ReadCloser `json:"-"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/events
func (c *Client) ListFineTuneEvents(ctx context.Context, req *ListFineTuneEventsRequest) (*ListFineTuneEventsResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.openai.com/v1/fine-tunes/"+req.ID+"/events", nil)
	if err != nil {
		return nil, err
	}

	if req.Stream {
		q := r.URL.Query()
		q.Set("stream", "true")
		r.URL.RawQuery = q.Encode()
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res ListFineTuneEventsResponse
	if !req.Stream {
		if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
			return nil, fmt.Errorf("failed to decode response: %w", err)
		}
	} else {
		res.Stream = resp.Body
	}

	return &res, nil
}

// https://platform.openai.com/docs/api-reference/fine-tunes/delete-model
type DeleteFineTuneModelRequest struct {
	// https://platform.openai.com/docs/api-reference/fine-tunes/delete-model#fine-tunes/delete-model-model
	//
	// Required.
	ID string `json:"model"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/delete-model
type DeleteFineTuneModelResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// https://platform.openai.com/docs/api-reference/fine-tunes/delete-model
func (c *Client) DeleteFineTuneModel(ctx context.Context, req *DeleteFineTuneModelRequest) (*DeleteFineTuneModelResponse, error) {
	r, err := http.NewRequestWithContext(ctx, http.MethodDelete, "https://api.openai.com/v1/fine-tunes/"+req.ID, nil)
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res DeleteFineTuneModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &res, nil
}

type ChatMessage struct {
	// Role is the role of the message, e.g. "user" or "bot".
	Role string `json:"role"`

	// Content is the text of the message.
	Content string `json:"content"`
}

// https://platform.openai.com/docs/api-reference/chat/create
type CreateChatRequest struct {
	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-model
	//
	// Required.
	Model string `json:"model"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-messages
	//
	// Required.
	Messages []ChatMessage `json:"messages"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
	//
	// Optional.
	Temperature float64 `json:"temperature,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
	//
	// Optional.
	TopP float64 `json:"top_p,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-n
	//
	// Optional.
	N int `json:"n,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-stream
	//
	// Optional.
	Stream bool `json:"stream,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop
	//
	// Optional.
	Stop []string `json:"stop,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
	//
	// Optional.
	MaxTokens int `json:"max_tokens,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-presence_penalty
	//
	// Optional.
	PresencePenalty float64 `json:"presence_penalty,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-frequency_penalty
	//
	// Optional.
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-logit_bias
	//
	// Optional.
	LogitBias map[string]float64 `json:"logit_bias,omitempty"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-user
	//
	// Optional.
	User string `json:"user,omitempty"`
}

// https://platform.openai.com/docs/api-reference/chat/create
type CreateChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	Model   string `json:"model"`
	Usage   struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Choices []struct {
		Message      ChatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
		Index        int         `json:"index"`
	} `json:"choices"`

	// https://platform.openai.com/docs/api-reference/chat/create#chat/create-stream
	Stream io.ReadCloser `json:"-"`
}

// https://platform.openai.com/docs/api-reference/chat/create
func (c *Client) CreateChat(ctx context.Context, req *CreateChatRequest) (*CreateChatResponse, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	r.Header.Add("Content-Type", "application/json")

	r.Header.Add("Authorization", "Bearer "+c.APIKey)

	if c.Organization != "" {
		r.Header.Set("OpenAI-Organization", c.Organization)
	}

	resp, err := c.HTTPClient.Do(r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		defer resp.Body.Close()
		return nil, fmt.Errorf("unexpected status code: %d: %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), body)
	}

	var res CreateChatResponse
	if !req.Stream {
		if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
			return nil, fmt.Errorf("failed to decode response: %w", err)
		}
		defer resp.Body.Close()
	} else {
		res.Stream = resp.Body
	}

	return &res, nil
}

// TODO:
// - https://beta.openai.com/docs/api-reference/images/create-edit
// - https://beta.openai.com/docs/api-reference/images/create-variation
