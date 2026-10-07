// Package safefile opens regular input files without following a replaced
// pathname into a symlink or blocking on a substituted FIFO/device.
package safefile

import "os"

// OpenRegular returns a checked file handle. Reads use this handle rather than
// reopening the pathname. Unix verifies the initial inode; Windows uses one
// atomic non-reparse open and pins the handle against deletion/rename.
func OpenRegular(path string) (*os.File, error) { return openRegular(path) }
