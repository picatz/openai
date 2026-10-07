//go:build unix

package safefile

import (
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestReplacedSymlinkAndFIFO(t *testing.T) {
	for _, kind := range []string{"symlink", "fifo"} {
		t.Run(kind, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "input")
			os.WriteFile(path, []byte("input"), 0600)
			expected, _ := os.Lstat(path)
			os.Remove(path)
			if kind == "symlink" {
				target := path + ".secret"
				os.WriteFile(target, []byte("unrelated private bytes"), 0600)
				if err := os.Symlink(target, path); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := unix.Mkfifo(path, 0600); err != nil {
					t.Fatal(err)
				}
			}
			done := make(chan error, 1)
			go func() {
				file, err := openChecked(path, expected)
				if file != nil {
					file.Close()
				}
				done <- err
			}()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("opened unsafe replacement")
				}
			case <-time.After(time.Second):
				t.Fatal("blocked opening substituted input")
			}
		})
	}
}

func TestRejectChangedRegularFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input")
	os.WriteFile(path, []byte("input"), 0600)
	expected, _ := os.Lstat(path)
	if err := os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(path, []byte("replacement"), 0600)
	if file, err := openChecked(path, expected); err == nil {
		file.Close()
		t.Fatal("opened replaced file")
	}
}
