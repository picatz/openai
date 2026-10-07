package safefile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenRegular(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input")
	if err := os.WriteFile(path, []byte("input"), 0600); err != nil {
		t.Fatal(err)
	}
	file, err := OpenRegular(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	data := make([]byte, 5)
	n, err := file.Read(data)
	if err != nil || n != 5 || string(data) != "input" {
		t.Fatalf("data=%q err=%v", data, err)
	}
}
