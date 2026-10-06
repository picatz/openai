package codex

import (
	"strings"
	"testing"
)

func TestBuildEnvironment(t *testing.T) {
	t.Setenv(internalOriginatorEnv, "")
	t.Setenv("OPENAI_BASE_URL", "existing")
	t.Setenv("CODEX_API_KEY", "existing")

	env := buildEnvironment("https://example.com", "test-key")

	assertEnvContains := func(key, expected string) {
		t.Helper()
		for _, entry := range env {
			if value, ok := strings.CutPrefix(entry, key+"="); ok {
				if value != expected {
					t.Fatalf("expected %s to be %q, got %q", key, expected, value)
				}
				return
			}
		}
		t.Fatalf("expected environment to contain %s", key)
	}

	assertEnvContains(internalOriginatorEnv, goSDKOriginator)
	assertEnvContains("OPENAI_BASE_URL", "https://example.com")
	assertEnvContains("CODEX_API_KEY", "test-key")
}
