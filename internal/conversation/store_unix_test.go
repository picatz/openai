//go:build !windows

package conversation

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPrivateFilesAndSymlinkRefusal(t *testing.T) {
	store := Store{Dir: filepath.Join(t.TempDir(), "sessions")}
	s := New(Chat, "local", "test")
	if err := store.Save(&s); err != nil {
		t.Fatal(err)
	}
	path, _ := store.path(s.ID)
	for _, tc := range []struct {
		path string
		mode os.FileMode
	}{{store.Dir, 0700}, {path, 0600}} {
		info, err := os.Stat(tc.path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != tc.mode {
			t.Errorf("mode=%o want=%o", info.Mode().Perm(), tc.mode)
		}
	}
	target := filepath.Join(t.TempDir(), "unrelated")
	os.WriteFile(target, []byte("private"), 0600)
	other := New(Chat, "local", "test")
	otherPath, _ := store.path(other.ID)
	if err := os.Symlink(target, otherPath); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(other.ID); err == nil {
		t.Fatal("followed symlink")
	}
	if err := store.Save(&other); err == nil {
		t.Fatal("overwrote symlink")
	}
	data, _ := os.ReadFile(target)
	if string(data) != "private" {
		t.Fatal("modified symlink target")
	}
}
