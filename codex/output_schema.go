package codex

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type outputSchemaFile struct {
	path    string
	cleanup func() error
}

func (f *outputSchemaFile) Path() string {
	if f == nil {
		return ""
	}
	return f.path
}

func (f *outputSchemaFile) Cleanup() error {
	if f == nil || f.cleanup == nil {
		return nil
	}
	return f.cleanup()
}

func createOutputSchemaFile(schema any) (*outputSchemaFile, error) {
	if schema == nil {
		return &outputSchemaFile{cleanup: func() error { return nil }}, nil
	}

	if err := ensureStructuredOutputIsJSONObject(schema); err != nil {
		return nil, err
	}

	dir, err := os.MkdirTemp("", "codex-output-schema-")
	if err != nil {
		return nil, err
	}

	cleanup := func() error {
		return os.RemoveAll(dir)
	}

	data, err := json.Marshal(schema)
	if err != nil {
		cleanup()
		return nil, err
	}

	path := filepath.Join(dir, "schema.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		cleanup()
		return nil, err
	}

	return &outputSchemaFile{path: path, cleanup: cleanup}, nil
}
