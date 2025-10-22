package codex

import (
	"os"
	"testing"
)

func TestNormalizeInputPrompt(t *testing.T) {
	prompt, images, err := normalizeInput(TextInput("hello world"))
	if err != nil {
		t.Fatalf("normalizeInput returned error: %v", err)
	}
	if prompt != "hello world" {
		t.Fatalf("expected prompt to be %q, got %q", "hello world", prompt)
	}
	if len(images) != 0 {
		t.Fatalf("expected no images, got %d", len(images))
	}
}

func TestNormalizeInputComposite(t *testing.T) {
	input := Input{
		Parts: []UserInput{
			TextPart("first"),
			LocalImagePart("path/to/image.png"),
			TextPart("second"),
		},
	}

	prompt, images, err := normalizeInput(input)
	if err != nil {
		t.Fatalf("normalizeInput returned error: %v", err)
	}
	expectedPrompt := "first\n\nsecond"
	if prompt != expectedPrompt {
		t.Fatalf("expected prompt %q, got %q", expectedPrompt, prompt)
	}
	if len(images) != 1 || images[0] != "path/to/image.png" {
		t.Fatalf("unexpected images: %#v", images)
	}
}

func TestNormalizeInputMissingImagePath(t *testing.T) {
	input := ComposeInput(LocalImagePart(""))
	if _, _, err := normalizeInput(input); err == nil {
		t.Fatal("expected an error for missing image path")
	}
}

func TestCreateOutputSchemaFile(t *testing.T) {
	t.Parallel()

	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}

	file, err := createOutputSchemaFile(schema)
	if err != nil {
		t.Fatalf("createOutputSchemaFile returned error: %v", err)
	}

	path := file.Path()
	if path == "" {
		t.Fatal("expected schema path to be populated")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read schema file: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("expected schema file to contain JSON data")
	}

	if err := file.Cleanup(); err != nil {
		t.Fatalf("cleanup returned error: %v", err)
	}

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected schema file to be removed, got err=%v", err)
	}
}

func TestCreateOutputSchemaFileRejectsNonObject(t *testing.T) {
	t.Parallel()

	if _, err := createOutputSchemaFile([]any{"not", "object"}); err == nil {
		t.Fatal("expected error for non-object schema")
	}
}
