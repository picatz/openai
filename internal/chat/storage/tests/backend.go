package tests

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/picatz/openai/internal/chat/storage"
	"github.com/shoenig/test/must"
)

// BackendSuite tests a backend implementation of the storage package, using
// the provided backend instance to perform the tests.
func BackendSuite(t *testing.T, backend storage.Backend[string, string]) {
	t.Helper()

	err := backend.Set(t.Context(), "hello", "world")
	must.NoError(t, err)

	value, ok, err := backend.Get(t.Context(), "hello")
	must.NoError(t, err)
	must.True(t, ok)
	must.Eq(t, "world", value)

	err = backend.Set(t.Context(), "hello again", "world2")
	must.NoError(t, err)

	value, ok, err = backend.Get(t.Context(), "hello again")
	must.NoError(t, err)
	must.True(t, ok)
	must.Eq(t, "world2", value)

	entries, next, err := backend.List(t.Context(), storage.PageSize(1), nil)
	must.NoError(t, err)
	must.NotNil(t, next)

	for key, value := range entries {
		must.Eq(t, "hello again", key)
		must.Eq(t, "world2", value)
	}

	entries, next, err = backend.List(t.Context(), nil, next)
	must.NoError(t, err)
	must.Nil(t, next)

	for key, value := range entries {
		must.Eq(t, "hello", key)
		must.Eq(t, "world", value)
	}
}

func BackendSuite_openai_chat_messages(t *testing.T, b storage.Backend[string, openai.ChatCompletionMessage]) {
	firstKey := "hello"
	secondKey := "hello again"

	firstMessage := openai.ChatCompletionMessage{Role: "user", Content: "world"}
	secondMessage := openai.ChatCompletionMessage{Role: "user", Content: "world2"}

	err := b.Set(t.Context(), firstKey, firstMessage)
	must.NoError(t, err)

	value, ok, err := b.Get(t.Context(), firstKey)
	must.NoError(t, err)
	must.True(t, ok)
	must.Eq(t, firstMessage.Content, value.Content)

	err = b.Set(t.Context(), secondKey, secondMessage)
	must.NoError(t, err)

	value, ok, err = b.Get(t.Context(), secondKey)
	must.NoError(t, err)
	must.True(t, ok)
	must.Eq(t, secondMessage.Content, value.Content)

	entries, next, err := b.List(t.Context(), storage.PageSize(1), nil)
	must.NoError(t, err)
	must.NotNil(t, next)

	for key, value := range entries {
		must.Eq(t, secondKey, key)
		must.Eq(t, secondMessage.Content, value.Content)
	}

	entries, next, err = b.List(t.Context(), nil, next)
	must.NoError(t, err)
	must.Nil(t, next)

	for key, value := range entries {
		must.Eq(t, firstKey, key)
		must.Eq(t, firstMessage.Content, value.Content)
	}
}
