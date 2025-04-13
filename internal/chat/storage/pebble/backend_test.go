package pebble_test

import (
	"testing"

	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/openai/openai-go"
	"github.com/picatz/openai/internal/chat/storage"
	backendPebble "github.com/picatz/openai/internal/chat/storage/pebble"
	"github.com/picatz/openai/internal/chat/storage/tests"
	"github.com/shoenig/test/must"
)

func TestBackend_dir(t *testing.T) {
	b, err := backendPebble.NewBackend(t.TempDir(), nil, &storage.JSONCodec[string, string]{})
	must.NoError(t, err)
	must.NotNil(t, b)

	tests.BackendSuite(t, b)
}

func TestBackend_mem_vfs(t *testing.T) {
	opts := &pebble.Options{
		FS: vfs.NewMem(),
	}

	b, err := backendPebble.NewBackend("", opts, &storage.JSONCodec[string, string]{})
	must.NoError(t, err)
	must.NotNil(t, b)

	tests.BackendSuite(t, b)
}

func TestBackend_mem_vfs_openai_chat_messages(t *testing.T) {
	codec := &storage.JSONCodec[string, openai.ChatCompletionMessage]{}

	opts := &pebble.Options{
		FS: vfs.NewMem(),
	}

	b, err := backendPebble.NewBackend("", opts, codec)
	must.NoError(t, err)
	must.NotNil(t, b)

	tests.BackendSuite_openai_chat_messages(t, b)
}
