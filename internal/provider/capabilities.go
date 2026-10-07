// Package provider describes explicitly declared API capabilities. It does not
// discover providers from a hostname or translate one API into another.
package provider

import "fmt"

// Endpoint identifies a wire protocol, not a model or provider brand.
type Endpoint string

const (
	ChatCompletions Endpoint = "chat"
	Responses       Endpoint = "responses"
	Decisions       Endpoint = "decisions"
)

// Path is the relative create path within an explicitly configured API base URL.
func (e Endpoint) Path() string {
	switch e {
	case ChatCompletions:
		return "chat/completions"
	case Responses:
		return "responses"
	case Decisions:
		return "decisions"
	default:
		return ""
	}
}

// Support distinguishes absent evidence from a documented unsupported feature.
// Unknown is deliberately the zero value; it must not silently become Supported.
type Support uint8

const (
	Unknown Support = iota
	Supported
	Unsupported
)

func (s Support) String() string {
	switch s {
	case Supported:
		return "supported"
	case Unsupported:
		return "unsupported"
	default:
		return "unknown"
	}
}

// EndpointCapabilities is endpoint- and model-specific. Tools and structured
// output still depend on the specific tool/schema; upstream validation is final.
// ToolChoice is separate because accepting tools does not imply tool_choice support.
// State means upstream storage/continuation, not client-side conversation replay.
type EndpointCapabilities struct {
	Endpoint         Support
	Streaming        Support
	Tools            Support
	ToolChoice       Support
	StructuredOutput Support
	State            Support
}

// Capabilities must be supplied from documented or explicitly tested behavior.
// Merely exposing /v1 does not establish any capability.
type Capabilities map[Endpoint]EndpointCapabilities

// For returns unknown capabilities for an undeclared endpoint. Decisions is
// unsupported by default: no other endpoint or structured-output feature implies it.
func (c Capabilities) For(endpoint Endpoint) EndpointCapabilities {
	if capability, ok := c[endpoint]; ok {
		return capability
	}
	if endpoint == Decisions {
		return EndpointCapabilities{Endpoint: Unsupported}
	}
	return EndpointCapabilities{}
}

// Check requires affirmative support. Unknown and unsupported remain distinct
// errors so callers can explain a missing declaration rather than claim support.
func Check(endpoint Endpoint, feature string, support Support) error {
	if support != Supported {
		return fmt.Errorf("%s %s is %s for this route", endpoint, feature, support)
	}
	return nil
}
