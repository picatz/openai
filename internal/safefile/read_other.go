//go:build !unix && !windows

package safefile

import (
	"fmt"
	"os"
)

func openRegular(path string) (*os.File, error) {
	return nil, fmt.Errorf("safe regular-file opening is not supported on this platform")
}
