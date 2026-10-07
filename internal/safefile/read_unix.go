//go:build unix

package safefile

import (
	"fmt"
	"golang.org/x/sys/unix"
	"os"
)

func openNative(path string) (*os.File, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), path), nil
}

func openRegular(path string) (*os.File, error) {
	expected, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !expected.Mode().IsRegular() {
		return nil, fmt.Errorf("input must be a regular file, not a symlink or device")
	}
	return openChecked(path, expected)
}
func openChecked(path string, expected os.FileInfo) (*os.File, error) {
	file, err := openNative(path)
	if err != nil {
		return nil, err
	}
	actual, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}
	if !actual.Mode().IsRegular() || !os.SameFile(expected, actual) {
		file.Close()
		return nil, fmt.Errorf("input file changed while opening")
	}
	return file, nil
}
