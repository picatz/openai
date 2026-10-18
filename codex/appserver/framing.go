package appserver

import (
	"bufio"
	"bytes"
	"io"
)

// frameReader never buffers more than the configured frame plus CRLF and its
// fixed read-ahead buffer. It accepts a final complete JSON frame without LF.
type frameReader struct {
	r     *bufio.Reader
	limit int
}

func newFrameReader(r io.Reader, limit int) *frameReader {
	return &frameReader{bufio.NewReaderSize(r, 4096), limit}
}
func (r *frameReader) next() ([]byte, error) {
	var frame []byte
	for {
		part, err := r.r.ReadSlice('\n')
		if len(part) > r.limit+2-len(frame) {
			return nil, ErrFrameLimit
		}
		frame = append(frame, part...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil && err != io.EOF {
			return nil, err
		}
		if err == io.EOF && len(frame) == 0 {
			return nil, io.EOF
		}
		frame = bytes.TrimSuffix(frame, []byte{'\n'})
		frame = bytes.TrimSuffix(frame, []byte{'\r'})
		if len(frame) > r.limit {
			return nil, ErrFrameLimit
		}
		return frame, nil
	}
}
