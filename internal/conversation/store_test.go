package conversation

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestStoreRoundtripAndConflict(t *testing.T) {
	store := Store{Dir: filepath.Join(t.TempDir(), "sessions")}
	s := New(Responses, "https://example.invalid/v1/", "test")
	s.Messages = []Message{{Role: "user", Content: "hello"}, {Role: "assistant", Content: "world"}}
	if err := store.Save(&s); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load(s.ID)
	if err != nil {
		t.Fatal(err)
	}
	stale := loaded.Clone()
	loaded.Messages = append(loaded.Messages, Message{Role: "user", Content: "next"})
	if err := store.Save(&loaded); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(&stale); !errors.Is(err, ErrConflict) {
		t.Fatalf("err=%v", err)
	}
	again, err := store.Load(s.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Messages) != 3 {
		t.Fatal("new history overwritten")
	}
	list, err := store.List()
	if err != nil || len(list) != 1 || list[0].Revision != 2 {
		t.Fatalf("list=%v err=%v", list, err)
	}
}
func TestStoreInvalidAndCorrupt(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	if _, err := store.Load("../../elsewhere"); err == nil {
		t.Fatal("accepted traversal")
	}
	s := New(Chat, "local", "test")
	path, _ := store.path(s.ID)
	os.WriteFile(path, []byte("broken"), 0600)
	if err := store.Save(&s); err == nil {
		t.Fatal("overwrote corrupt history")
	}
	got, _ := os.ReadFile(path)
	if string(got) != "broken" {
		t.Fatal("corrupt history changed")
	}
}
func TestStoreLockPreservesHistory(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	s := New(Chat, "local", "test")
	if err := store.Save(&s); err != nil {
		t.Fatal(err)
	}
	path, _ := store.path(s.ID)
	before, _ := os.ReadFile(path)
	os.WriteFile(path+".lock", nil, 0600)
	if err := store.Save(&s); err == nil {
		t.Fatal("ignored lock")
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("history changed")
	}
}
