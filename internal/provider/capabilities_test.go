package provider

import "testing"

func TestUnknownIsNotUnsupportedOrSupported(t *testing.T) {
	var capabilities Capabilities
	for _, endpoint := range []Endpoint{ChatCompletions, Responses} {
		got := capabilities.For(endpoint)
		if got != (EndpointCapabilities{}) || got.Endpoint != Unknown {
			t.Fatalf("%s = %+v", endpoint, got)
		}
		if Check(endpoint, "endpoint", got.Endpoint) == nil {
			t.Fatalf("undeclared %s was accepted", endpoint)
		}
	}
	if got := capabilities.For(Decisions).Endpoint; got != Unsupported {
		t.Fatalf("Decisions default = %s", got)
	}
	capabilities = Capabilities{Responses: {Endpoint: Supported, Streaming: Unsupported}}
	if Check(Responses, "endpoint", capabilities.For(Responses).Endpoint) != nil {
		t.Fatal("declared endpoint was rejected")
	}
	if got := capabilities.For(Responses); got.Tools != Unknown || got.State != Unknown || got.Streaming != Unsupported {
		t.Fatalf("endpoint support implied unrelated capabilities: %+v", got)
	}
}

func TestEndpointPaths(t *testing.T) {
	for endpoint, want := range map[Endpoint]string{ChatCompletions: "chat/completions", Responses: "responses", Decisions: "decisions", "invalid": ""} {
		if got := endpoint.Path(); got != want {
			t.Errorf("%q path = %q, want %q", endpoint, got, want)
		}
	}
}
