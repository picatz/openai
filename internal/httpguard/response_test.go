package httpguard

import (
	"io"
	"strings"
	"testing"
)

func TestBoundedBody(t *testing.T) {
	for _, tc := range []struct {
		input string
		limit int64
		fail  bool
	}{{"abc", 3, false}, {"abcd", 3, true}, {"", 3, false}} {
		b := &boundedBody{ReadCloser: io.NopCloser(strings.NewReader(tc.input)), remaining: tc.limit, limit: tc.limit}
		data, err := io.ReadAll(b)
		if (err != nil) != tc.fail || len(data) > int(tc.limit) {
			t.Fatalf("data=%q err=%v", data, err)
		}
	}
}
