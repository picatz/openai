package appserver

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
)

func TestEnvelopeValidation(t *testing.T) {
	valid := []string{`{"id":9223372036854775807,"result":null}`, `{"id":-9223372036854775808,"result":false}`, `{"id":"1","result":{}}`, `{"method":"future","futureField":true}`, `{"id":1,"method":"x","params":null}`, `{"id":-1,"error":{"code":-32601,"message":"no","data":{"future":true}}}`}
	for _, s := range valid {
		if _, err := parseEnvelope([]byte(s)); err != nil {
			t.Errorf("valid %q: %v", s, err)
		}
	}
	invalid := []string{`null`, `[]`, `1`, `{}`, `{"method":null}`, `{"method":""}`, `{"id":null,"method":"x"}`, `{"id":1.0,"result":null}`, `{"id":1e0,"result":null}`, `{"id":9223372036854775808,"result":null}`, `{"id":-9223372036854775809,"result":null}`, `{"id":true,"result":null}`, `{"id":1}`, `{"result":null}`, `{"id":1,"result":null,"error":{}}`, `{"id":1,"method":"x","result":null}`, `{"id":1,"result":null,"params":{}}`, `{"id":1,"error":null}`, `{"id":1,"error":{"code":1.1,"message":"x"}}`, `{"id":1,"error":{"code":1,"message":null}}`, `{"id":1,"error":{"code":1}}`, `{"id":1,"result":null,"id":2}`, `{"id":1,"result":{"x":1,"\u0078":2}}`, `{"method":"x"} {}`, `{"method":"x"`, string([]byte{'{', '"', 'm', 'e', 't', 'h', 'o', 'd', '"', ':', '"', 0xff, '"', '}'})}
	for _, s := range invalid {
		if _, err := parseEnvelope([]byte(s)); !errors.Is(err, ErrProtocol) {
			t.Errorf("invalid %q accepted: %v", s, err)
		}
	}
	if err := validateJSON([]byte(strings.Repeat("[", 66) + "0" + strings.Repeat("]", 66))); err == nil {
		t.Fatal("deep nesting accepted")
	}
	a, _ := parseID([]byte(`1`))
	b, _ := parseID([]byte(`"1"`))
	if a.key == b.key {
		t.Fatal("ID types collapsed")
	}
}
func TestOwnedEnvelope(t *testing.T) {
	raw := []byte(`{"method":"future","params":{"value":1},"extra":true}`)
	e, err := parseEnvelope(raw)
	if err != nil {
		t.Fatal(err)
	}
	for i := range raw {
		raw[i] = 'x'
	}
	if !json.Valid(e.raw) || string(e.params) != `{"value":1}` {
		t.Fatal("payload aliases read buffer")
	}
}
func TestFrames(t *testing.T) {
	for _, suffix := range []string{"", "\n", "\r\n"} {
		r := newFrameReader(strings.NewReader("1234"+suffix), 4)
		b, e := r.next()
		if e != nil || string(b) != "1234" {
			t.Fatalf("suffix %q: %q %v", suffix, b, e)
		}
		_, e = r.next()
		if e != io.EOF {
			t.Fatal(e)
		}
	}
	for _, suffix := range []string{"", "\n", "\r\n"} {
		if _, e := newFrameReader(strings.NewReader("12345"+suffix), 4).next(); !errors.Is(e, ErrFrameLimit) {
			t.Fatal(e)
		}
	}
	big := strings.Repeat("a", 70<<10)
	r := newFrameReader(oneByteReader{strings.NewReader(big + "\r\n")}, len(big))
	b, e := r.next()
	if e != nil || string(b) != big {
		t.Fatal("large/fragmented frame", e)
	}
}

type oneByteReader struct{ io.Reader }

func (r oneByteReader) Read(p []byte) (int, error) { return r.Reader.Read(p[:1]) }
func TestPinnedFixtures(t *testing.T) {
	f, err := os.Open("testdata/wire.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scan := bufio.NewScanner(f)
	n := 0
	for scan.Scan() {
		if _, err := parseEnvelope(scan.Bytes()); err != nil {
			t.Fatalf("fixture %d: %v", n, err)
		}
		n++
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	if n != 13 {
		t.Fatal(n)
	}
}
func FuzzEnvelope(f *testing.F) {
	for _, s := range []string{`{"method":"future","params":null}`, `{"id":9223372036854775807,"result":null}`, `{"id":"go:1","error":{"code":-1,"message":"x"}}`} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<20 {
			return
		}
		e, err := parseEnvelope(b)
		if err == nil {
			if !json.Valid(e.raw) || !bytes.Equal(e.raw, b) {
				t.Fatal("lost valid owned frame")
			}
			for i := range b {
				b[i] = 0
			}
			if !json.Valid(e.raw) {
				t.Fatal("aliased raw")
			}
		}
	})
}
func FuzzFraming(f *testing.F) {
	f.Add([]byte("abc\n"), uint16(3))
	f.Add([]byte("abc\r\n"), uint16(3))
	f.Fuzz(func(t *testing.T, b []byte, n uint16) {
		if len(b) > 1<<20 {
			return
		}
		limit := int(n) + 1
		r := newFrameReader(bytes.NewReader(b), limit)
		for i := 0; i <= len(b)+1; i++ {
			frame, err := r.next()
			if len(frame) > limit {
				t.Fatal("frame limit exceeded")
			}
			if err != nil {
				return
			}
		}
		t.Fatal("reader made no progress")
	})
}

func TestUnicodeEscapes(t *testing.T) {
	for _, s := range []string{`{"id":"\ud800","result":null}`, `{"id":"\udfff","result":null}`, `{"method":"x","params":{"\ud800":1}}`, `{"method":"x","params":["\ud800\u0041"]}`, `{"method":"x","params":"\uDBFF\uDFFF\uDC00"}`} {
		if _, e := parseEnvelope([]byte(s)); e == nil {
			t.Errorf("unpaired surrogate accepted: %s", s)
		}
	}
	for _, s := range []string{`{"id":"\ud83d\ude00","result":null}`, `{"id":"\\ud800","result":null}`, `{"method":"x","params":"\ufffd"}`} {
		if _, e := parseEnvelope([]byte(s)); e != nil {
			t.Errorf("valid escape rejected: %s %v", s, e)
		}
	}
	a, _ := parseEnvelope([]byte(`{"id":"\ud83d\ude00","result":null}`))
	b, _ := parseEnvelope([]byte(`{"id":"😀","result":null}`))
	if a.id.key != b.id.key {
		t.Fatal("equivalent Unicode IDs differ")
	}
}

type countingRepeater struct{ n int }

func (r *countingRepeater) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'x'
	}
	r.n += len(p)
	return len(p), nil
}
func TestFramingRejectsUnboundedInputEarly(t *testing.T) {
	input := &countingRepeater{}
	_, err := newFrameReader(input, 8192).next()
	if !errors.Is(err, ErrFrameLimit) {
		t.Fatal(err)
	}
	if input.n > 8192+4096 {
		t.Fatal("excessive read-ahead", input.n)
	}
	if _, err := parseEnvelope([]byte(`{"method":"unterminated"`)); !errors.Is(err, ErrProtocol) {
		t.Fatal(err)
	}
}
