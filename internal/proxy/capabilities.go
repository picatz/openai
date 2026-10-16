package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/picatz/openai/internal/provider"
)

var errInvalidParameter = errors.New("invalid request parameter")

func checkCapabilities(endpoint provider.Endpoint, capability provider.EndpointCapabilities, object map[string]json.RawMessage) error {
	if err := provider.Check(endpoint, "endpoint", capability.Endpoint); err != nil {
		return err
	}
	stream, err := boolParameter(object, "stream")
	if err != nil {
		return err
	}
	if stream {
		if err := provider.Check(endpoint, "streaming", capability.Streaming); err != nil {
			return err
		}
	}
	if nonempty(object["tools"]) || (endpoint == provider.ChatCompletions && nonempty(object["functions"])) {
		if err := provider.Check(endpoint, "tools", capability.Tools); err != nil {
			return err
		}
	}
	if nonempty(object["tool_choice"]) || (endpoint == provider.ChatCompletions && nonempty(object["function_call"])) {
		if err := provider.Check(endpoint, "tool choice", capability.ToolChoice); err != nil {
			return err
		}
	}
	var format json.RawMessage
	switch endpoint {
	case provider.ChatCompletions:
		format = object["response_format"]
	case provider.Responses:
		if !absent(object["text"]) {
			text, err := envelopeObject(object["text"], false, "format")
			if err != nil {
				return fmt.Errorf("%w: text must be an object with one unambiguous format parameter", errInvalidParameter)
			}
			format = text["format"]
		}
	}
	structured, err := structuredFormat(format)
	if err != nil {
		return err
	}
	if structured {
		if err := provider.Check(endpoint, "structured output", capability.StructuredOutput); err != nil {
			return err
		}
	}
	store, err := boolParameter(object, "store")
	if err != nil {
		return err
	}
	state := store
	if endpoint == provider.Responses {
		background, err := boolParameter(object, "background")
		if err != nil {
			return err
		}
		state = state || background || nonDefault(object["previous_response_id"], `""`) || nonDefault(object["conversation"], `""`)
	}
	if state {
		if err := provider.Check(endpoint, "state", capability.State); err != nil {
			return err
		}
	}
	// Responses defaults can persist server-side state. A stateless/unknown
	// route must opt out explicitly; the proxy never silently rewrites store.
	if endpoint == provider.Responses && capability.State != provider.Supported && !bytes.Equal(bytes.TrimSpace(object["store"]), []byte("false")) {
		return fmt.Errorf("responses state is %s for this route; explicitly set store:false", capability.State)
	}
	return nil
}

func boolParameter(object map[string]json.RawMessage, key string) (bool, error) {
	var value bool
	if raw, exists := object[key]; exists {
		if err := json.Unmarshal(raw, &value); err != nil {
			return false, fmt.Errorf("%w: %s must be a boolean", errInvalidParameter, key)
		}
	}
	return value, nil
}

func nonempty(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) != 0 && !bytes.Equal(raw, []byte("null")) && !bytes.Equal(raw, []byte("[]"))
}

func nonDefault(raw json.RawMessage, defaultValue string) bool {
	return nonempty(raw) && !bytes.Equal(bytes.TrimSpace(raw), []byte(defaultValue))
}

func absent(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) == 0 || bytes.Equal(raw, []byte("null"))
}

func structuredFormat(raw json.RawMessage) (bool, error) {
	if absent(raw) {
		return false, nil
	}
	format, err := envelopeObject(raw, false, "type")
	if err != nil {
		return false, fmt.Errorf("%w: format must be an object with one unambiguous type parameter", errInvalidParameter)
	}
	var kind string
	if value, ok := format["type"]; ok {
		if err := json.Unmarshal(value, &kind); err != nil {
			return false, fmt.Errorf("%w: format type must be a string", errInvalidParameter)
		}
	}
	// Any non-default format needs an explicit declaration, including future
	// formats. The upstream, not this routing envelope, validates its schema.
	// Decode only the exact key: struct decoding would fold case and could
	// disagree with a case-sensitive provider about which format was requested.
	return kind != "text", nil
}
