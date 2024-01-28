package main

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// TODO: make cross platform, macos only for now
func readClipboard() (string, error) {
	if runtime.GOOS != "darwin" {
		return "", fmt.Errorf("readClipboard: unsupported platform: %s", runtime.GOOS)
	}

	cmd := exec.Command("pbpaste")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func writeClipboard(s string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("writeClipboard: unsupported platform: %s", runtime.GOOS)
	}

	cmd := exec.Command("pbcopy")
	cmd.Stdin = strings.NewReader(s)
	return cmd.Run()
}
