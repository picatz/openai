package codex

import (
	"encoding/json"
	"os"
	"testing"
)

type countingSchema struct{ count int }

func (s *countingSchema) MarshalJSON() ([]byte, error) {
	s.count++
	return []byte(`{"type":"object"}`), nil
}

func TestOutputSchemaMarshaledOnceAndCleaned(t *testing.T) {
	dir := schemaTempDir(t)
	schema := &countingSchema{}
	file, err := createOutputSchemaFile(schema)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Cleanup()
	raw, err := os.ReadFile(file.Path())
	if err != nil || !json.Valid(raw) || schema.count != 1 {
		t.Fatalf("schema = %s, count = %d, err = %v", raw, schema.count, err)
	}
	if err := file.Cleanup(); err != nil {
		t.Fatal(err)
	}
	assertSchemaCleaned(t, dir)
}

func TestOutputSchemaRejectsNonObjects(t *testing.T) {
	dir := schemaTempDir(t)
	for _, value := range []any{[]string{}, "schema", 1, true, json.RawMessage(`null`), (map[string]any)(nil), make(chan bool)} {
		if _, err := createOutputSchemaFile(value); err == nil {
			t.Fatalf("accepted %T", value)
		}
		assertSchemaCleaned(t, dir)
	}
	file, err := createOutputSchemaFile(nil)
	if err != nil || file.Path() != "" {
		t.Fatalf("nil schema = %+v, %v", file, err)
	}
}
