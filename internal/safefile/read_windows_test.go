//go:build windows

package safefile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenHandlePinsWindowsPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input")
	os.WriteFile(path, []byte("input"), 0600)
	file, err := OpenRegular(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if err := os.Rename(path, path+".moved"); err == nil {
		t.Fatal("opened input was replaceable while reading")
	}
}
func TestWindowsRejectsReparsePoint(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	link := filepath.Join(dir, "link")
	os.WriteFile(target, []byte("private"), 0600)
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink creation is unavailable: %v", err)
	}
	if f, err := OpenRegular(link); err == nil {
		f.Close()
		t.Fatal("followed reparse point")
	}
}
