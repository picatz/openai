package memory_test

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/picatz/openai/internal/chat/storage/memory"
	"github.com/picatz/openai/internal/chat/storage/tests"
)

func TestBackend(t *testing.T) {
	tests.BackendSuite(t, memory.NewBackend[string, string]())
	tests.BackendSuite_openai_chat_messages(t, memory.NewBackend[string, openai.ChatCompletionMessage]())
}
