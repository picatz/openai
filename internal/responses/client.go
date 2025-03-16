package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// DefaultBaseURL is the default base URL for the OpenAI API.
const DefaultBaseURL = "https://api.openai.com/v1"

// Client represents an OpenAI API client for the responses endpoint.
type Client struct {
	// HTTPClient is the HTTP client used to communicate with the API.
	// If nil, http.DefaultClient is used.
	HTTPClient *http.Client

	// APIKey is your OpenAI API key.
	APIKey string

	// BaseURL is the base URL for the OpenAI API.
	BaseURL string
}

// NewClient creates a new Client for the OpenAI responses API.
func NewClient(apiKey string, httpClient *http.Client) *Client {
	return &Client{
		HTTPClient: httpClient,
		APIKey:     apiKey,
		BaseURL:    DefaultBaseURL,
	}
}

// Request represents the request payload for the responses API.
type Request struct {
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-model
	Model string `json:"model"`

	// Text, image, or file inputs to the model, used to generate a response.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-input
	Input RequestInput `json:"input"`

	// Inserts a system (or developer) message as the first item in the model's context.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-instructions
	Instructions string `json:"instructions,omitzero"`

	// An upper bound for the number of tokens that can be generated for a response, including
	// visible output tokens and reasoning tokens.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-max_output_tokens
	MaxOutputTokens uint64 `json:"max_output_tokens,omitzero"`

	// Set of 16 key-value pairs that can be attached to an object. This can be useful for storing
	// additional information about the object in a structured format, and querying for objects via
	// API or the dashboard.
	//
	// Keys are strings with a maximum length of 64 characters. Values are strings with a maximum
	// length of 512 characters.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-metadata
	Metadata map[string]string `json:"metadata,omitempty"`

	// Configuration options for reasoning models.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-reasoning
	Reasoning *RequestResoning `json:"reasoning,omitempty"`

	// Whether to store the generated model response for later retrieval via API (default: true).
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-store
	Store bool `json:"store,omitzero"`

	// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
	// output more random, while lower values like 0.2 will make it more focused and deterministic.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-temperature
	Temperature *float64 `json:"temperature,omitempty"`

	// An alternative to sampling with temperature, called nucleus sampling, where the model considers
	// the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising
	// the top 10% probability mass are considered.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-top_p
	TopP *float64 `json:"top_p,omitempty"`

	// The truncation strategy to use for the model response.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-truncation
	Truncation RequestTruncation `json:"truncation,omitzero"`

	// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-user
	User *string `json:"user,omitempty"`

	// An array of tools the model may call while generating a response. You can specify which
	// tool to use by setting the tool_choice parameter.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools
	Tools RequestTools `json:"tools,omitempty"`

	// How the model should select which tool (or tools) to use when generating a response.
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-tool_choice
	ToolChoice RequestToolChoice `json:"tool_choice,omitzero"`

	// The unique ID of the previous response to the model. Use this to create multi-turn [conversations].
	//
	// [conversations]: https://platform.openai.com/docs/guides/conversation-state?api-mode=responses
	//
	// https://platform.openai.com/docs/api-reference/responses/create#responses-create-previous_response_id
	PreviousResponseID string `json:"previous_response_id,omitzero"`
}

// RequestToolChoice represents the tool choice for the model.
type RequestToolChoice string

const (
	// RequestToolChoiceAuto means the model can pick between generating a message or calling one or more tools.
	RequestToolChoiceAuto RequestToolChoice = "auto"

	// RequestToolChoiceNone means the model will not call any tool and instead generates a message.
	RequestToolChoiceNone RequestToolChoice = "none"

	// RequestToolChoiceRequired means the model must call one or more tools.
	RequestToolChoiceRequired RequestToolChoice = "required"
)

type RequestTools []requestTool

// https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools
type requestTool interface {
	isRequestTool()
}

// JSONSchema is a JSON Schema.
//
// https://json-schema.org/understanding-json-schema/reference/index.html
type JSONSchema struct {
	// Type is the type of the schema.
	Type string `json:"type,omitempty"`

	// Description is the description of the schema.
	Description string `json:"description,omitempty"`

	// Properties is the properties of the schema.
	Properties map[string]*JSONSchema `json:"properties,omitempty"`

	// Required is the required properties of the schema.
	Required []string `json:"required,omitempty"`

	// Enum is the enum of the schema.
	Enum []string `json:"enum,omitempty"`

	// Items is the items of the schema.
	Items *JSONSchema `json:"items,omitempty"`

	// AdditionalProperties is the additional properties of the schema.
	AdditionalProperties *JSONSchema `json:"additionalProperties,omitempty"`

	// Ref is the ref of the schema.
	Ref string `json:"$ref,omitempty"`

	// AnyOf is the anyOf of the schema.
	AnyOf []*JSONSchema `json:"anyOf,omitempty"`

	// AllOf is the allOf of the schema.
	AllOf []*JSONSchema `json:"allOf,omitempty"`

	// OneOf is the oneOf of the schema.
	OneOf []*JSONSchema `json:"oneOf,omitempty"`

	// Default is the default of the schema.
	Default any `json:"default,omitempty"`

	// Pattern is the pattern of the schema.
	Pattern string `json:"pattern,omitempty"`

	// MinItems is the minItems of the schema.
	MinItems int `json:"minItems,omitempty"`

	// MaxItems is the maxItems of the schema.
	MaxItems int `json:"maxItems,omitempty"`

	// UniqueItems is the uniqueItems of the schema.
	UniqueItems bool `json:"uniqueItems,omitempty"`

	// MultipleOf is the multipleOf of the schema.
	MultipleOf int `json:"multipleOf,omitempty"`

	// Min is the minimum of the schema.
	Min int `json:"min,omitempty"`

	// Max is the maximum of the schema.
	Max int `json:"max,omitempty"`

	// ExclusiveMin is the exclusiveMinimum of the schema.
	ExclusiveMin bool `json:"exclusiveMinimum,omitempty"`

	// ExclusiveMax is the exclusiveMaximum of the schema.
	ExclusiveMax bool `json:"exclusiveMaximum,omitempty"`
}

type RequestToolFunction struct {
	// The name of the function to call.
	Name string `json:"name"`

	// A JSON schema object describing the parameters of the function.
	Parameters *JSONSchema `json:"parameters,omitempty"`

	// Whether to enforce strict parameter validation (default: true).
	Strict bool `json:"strict,omitzero"`

	// A description of the function. Used by the model to determine whether or not to call the function.
	Description string `json:"description,omitzero"`
}

func (RequestToolFunction) isRequestTool() {}

func (tf RequestToolFunction) MarshalJSON() ([]byte, error) {
	type alias RequestToolFunction
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "function",
		alias: (alias)(tf),
	})
}

type RequestToolWebSearchPreview struct{}

func (RequestToolWebSearchPreview) isRequestTool() {}

func (RequestToolWebSearchPreview) MarshalJSON() ([]byte, error) {
	type alias RequestToolWebSearchPreview
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "web_search_preview",
		alias: (alias)(RequestToolWebSearchPreview{}),
	})
}

type RequestTruncation string

const (
	RequestTruncationAuto     RequestTruncation = "auto"
	RequestTruncationDisabled RequestTruncation = "disabled"
)

func (tr RequestTruncation) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(tr))
}

// https://platform.openai.com/docs/api-reference/responses/create#responses-create-reasoning
type RequestResoning struct {
	// O-series models only.
	Effort string `json:"effort,omitzero"`

	// Computer use preview only.
	GenerateSummary bool `json:"generate_summary,omitzero"`
}

type RequestInput interface {
	isRequestInput()
}

// https://platform.openai.com/docs/api-reference/responses/list
type ItemList []Item

func (ItemList) isRequestInput()   {}
func (ItemList) isMessageContent() {}

func (il *ItemList) UnmarshalJSON(b []byte) error {
	var items []map[string]any

	if err := json.Unmarshal(b, &items); err != nil {
		return err
	}

	for _, item := range items {
		itemType, ok := item["type"].(string)
		if !ok {
			return fmt.Errorf("missing type field in item: %v", item)
		}
		switch itemType {
		case "message":
			b, err := json.Marshal(item)
			if err != nil {
				return fmt.Errorf("failed to marshal item data: %w", err)
			}

			var msg Message
			if err := json.Unmarshal(b, &msg); err != nil {
				return err
			}
			*il = append(*il, msg)
		default:
			return fmt.Errorf("unknown item type: %s", itemType)
		}
	}

	return nil
}

type Item interface {
	isInputItem()
}

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
)

type MessageContent interface {
	isMessageContent()
}

type Text string

func (Text) isMessageContent() {}
func (Text) isRequestInput()   {}

type Message struct {
	Role    Role           `json:"role"`
	Content MessageContent `json:"content"`
	Status  string         `json:"status,omitzero"`
}

func (Message) isInputItem() {}

func (ii Message) MarshalJSON() ([]byte, error) {
	type alias Message
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "message",
		alias: (alias)(ii),
	})
}

func (m *Message) UnmarshalJSON(b []byte) error {
	type alias Message
	aux := &struct {
		Content json.RawMessage `json:"content"`
		*alias
	}{
		alias: (*alias)(m),
	}

	// Unmarshal into the auxiliary structure
	if err := json.Unmarshal(b, aux); err != nil {
		return err
	}

	// If no content is provided, nothing to do
	if len(aux.Content) == 0 {
		return nil
	}

	switch aux.Content[0] {
	case '"':
		var text Text
		if err := json.Unmarshal(aux.Content, &text); err != nil {
			return err
		}
		m.Content = text
		return nil
	case '[':
		var rawArr []json.RawMessage
		if err := json.Unmarshal(aux.Content, &rawArr); err != nil {
			return err
		}
		var contents ItemList
		for _, raw := range rawArr {
			var typeHolder struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(raw, &typeHolder); err != nil {
				return err
			}
			switch typeHolder.Type {
			case "input_text":
				var inputText InputText
				if err := json.Unmarshal(raw, &inputText); err != nil {
					return err
				}
				contents = append(contents, inputText)
			case "input_image":
				var inputImage InputImage
				if err := json.Unmarshal(raw, &inputImage); err != nil {
					return err
				}
				contents = append(contents, inputImage)
			case "input_file":
				var inputFile InputFile
				if err := json.Unmarshal(raw, &inputFile); err != nil {
					return err
				}
				contents = append(contents, inputFile)
			default:
				return fmt.Errorf("unknown message content type: %s", typeHolder.Type)
			}

			m.Content = contents
		}
	default:
		return fmt.Errorf("unknown message content type: %q: %q", aux.Content[0], aux.Content)
	}

	return nil
}

type OutputMessage struct {
	Role    string                 `json:"role"`
	Content []OutputMessageContent `json:"content"`
	ID      string                 `json:"id"`
	Status  string                 `json:"status"`
}

func (OutputMessage) isInputItem() {}

func (ii OutputMessage) MarshalJSON() ([]byte, error) {
	type alias OutputMessage
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "message",
		alias: (alias)(ii),
	})
}

type OutputMessageContent interface {
	isOutputMessage()
}

type OutputText struct {
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

func (OutputText) isOutputMessage() {}

func (outputText OutputText) MarshalJSON() ([]byte, error) {
	type alias OutputText
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "output_text",
		alias: (alias)(outputText),
	})
}

type OutputRefusal struct {
	Refusal string `json:"refusal"`
}

func (OutputRefusal) isOutputMessage() {}

func (of OutputRefusal) MarshalJSON() ([]byte, error) {
	type alias OutputRefusal
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "refusal",
		alias: (alias)(of),
	})
}

type InputText struct {
	Text string `json:"text"`
}

func (InputText) isMessageContent() {}
func (InputText) isInputItem()      {}

func (it InputText) MarshalJSON() ([]byte, error) {
	type Alias InputText
	return json.Marshal(struct {
		Type string `json:"type"`
		*Alias
	}{
		Type:  "input_text",
		Alias: (*Alias)(&it),
	})
}

type InputImage struct {
	Detail   string `json:"detail"`
	FileID   string `json:"file_id,omitzero"`
	ImageURL string `json:"image_url,omitzero"`
}

// func (InputImage) isMessageContent() {}
func (InputImage) isInputItem() {}

func (ii InputImage) MarshalJSON() ([]byte, error) {
	type alias InputImage
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "input_image",
		alias: (alias)(ii),
	})
}

type InputFile struct {
	FileID   string `json:"file_id,omitzero"`
	FileData string `json:"file_data,omitzero"`
	Filename string `json:"filename,omitzero"`
}

// func (InputFile) isMessageContent() {}
func (InputFile) isInputItem() {}

func (ii InputFile) MarshalJSON() ([]byte, error) {
	type alias InputFile
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{
		Type:  "input_file",
		alias: (alias)(ii),
	})
}

// Response represents a complete response returned by the responses API.
//
// https://platform.openai.com/docs/api-reference/responses/object
type Response struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	CreatedAt         int    `json:"created_at"`
	Status            string `json:"status"`
	IncompleteDetails any    `json:"incomplete_details"`
	Instructions      any    `json:"instructions"`
	MaxOutputTokens   any    `json:"max_output_tokens"`
	Model             string `json:"model"`
	Output            []struct {
		Type    string `json:"type"`
		ID      string `json:"id"`
		Status  string `json:"status"`
		Role    string `json:"role"`
		Content []struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Annotations []any  `json:"annotations"`
		} `json:"content"`
		Name     string `json:"name"`
		CallID   string `json:"call_id"`
		Aruments string `json:"arguments"`
	} `json:"output"`
	ParallelToolCalls  bool `json:"parallel_tool_calls"`
	PreviousResponseID any  `json:"previous_response_id"`
	Reasoning          struct {
		Effort  any `json:"effort"`
		Summary any `json:"summary"`
	} `json:"reasoning"`
	Store       bool    `json:"store"`
	Temperature float64 `json:"temperature"`
	Text        struct {
		Format struct {
			Type string `json:"type"`
		} `json:"format"`
	} `json:"text"`
	ToolChoice string  `json:"tool_choice"`
	Tools      []any   `json:"tools"`
	TopP       float64 `json:"top_p"`
	Truncation string  `json:"truncation"`
	Usage      struct {
		InputTokens        int `json:"input_tokens"`
		InputTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputTokens        int `json:"output_tokens"`
		OutputTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
	User     any `json:"user"`
	Metadata struct {
	} `json:"metadata"`
}

func handleErrorResponse(resp *http.Response) error {
	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			} `json:"error"`
		}

		err := unmarshalJSON(resp.Body, &errResp)
		if err != nil {
			return fmt.Errorf("failed to unmarshal error response: %w", err)
		}

		return fmt.Errorf("API error: %s: %s: %s", errResp.Error.Code, errResp.Error.Type, errResp.Error.Message)
	}
	return nil
}

// unmarshalJSON reads the response body and unmarshals it into the provided result struct.
func unmarshalJSON(r io.Reader, result any) error {
	b, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	err = json.Unmarshal(b, result)
	if err != nil {
		return err
	}
	return nil
}

// https://platform.openai.com/docs/api-reference/responses/create
func (c *Client) Create(ctx context.Context, reqData Request) (*Response, error) {
	url := fmt.Sprintf("%s/responses", c.BaseURL)

	body, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, handleErrorResponse(resp)
	}

	var apiResp Response
	err = unmarshalJSON(resp.Body, &apiResp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal API response: %w", err)
	}

	return &apiResp, nil
}

// https://platform.openai.com/docs/api-reference/responses/get
func (c *Client) Get(ctx context.Context, responseID string) (*Response, error) {
	url := fmt.Sprintf("%s/responses/%s", c.BaseURL, responseID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, handleErrorResponse(resp)
	}

	var apiResp Response
	err = unmarshalJSON(resp.Body, &apiResp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal API response: %w", err)
	}

	return &apiResp, nil
}

// https://platform.openai.com/docs/api-reference/responses/delete
func (c *Client) Delete(ctx context.Context, responseID string) error {
	url := fmt.Sprintf("%s/responses/%s", c.BaseURL, responseID)

	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		return handleErrorResponse(resp)
	}

	return nil
}

type GetInputItemsOptions struct {
	After  string
	Before string
	Limit  int
	Order  string
}

type ResponseInputItems struct {
	FirstID string   `json:"first_id"`
	LastID  string   `json:"last_id"`
	HasMore bool     `json:"has_more"`
	Data    ItemList `json:"data"`
}

// https://platform.openai.com/docs/api-reference/responses/input-items
func (c *Client) GetInputItems(ctx context.Context, responseID string, opts *GetInputItemsOptions) (*ResponseInputItems, error) {
	url := fmt.Sprintf("%s/responses/%s/input_items", c.BaseURL, responseID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	// Set query parameters
	if opts != nil {
		query := req.URL.Query()
		if opts.After != "" {
			query.Add("after", opts.After)
		}
		if opts.Before != "" {
			query.Add("before", opts.Before)
		}
		if opts.Limit > 0 {
			query.Add("limit", fmt.Sprintf("%d", opts.Limit))
		}
		if opts.Order != "" {
			query.Add("order", opts.Order)
		}
		req.URL.RawQuery = query.Encode()
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, handleErrorResponse(resp)
	}

	var apiResp ResponseInputItems
	err = unmarshalJSON(resp.Body, &apiResp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal API response: %w", err)
	}

	return &apiResp, nil
}
