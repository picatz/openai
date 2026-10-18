package appserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"unicode/utf8"
)

var (
	ErrClosed         = errors.New("appserver: connection closed")
	ErrProtocol       = errors.New("appserver: invalid protocol message")
	ErrFrameLimit     = errors.New("appserver: frame byte limit exceeded")
	ErrEventLimit     = errors.New("appserver: event queue limit exceeded; stream incomplete")
	ErrWriteLimit     = errors.New("appserver: write queue limit exceeded")
	ErrPendingLimit   = errors.New("appserver: pending call limit reached")
	ErrAbandonedLimit = errors.New("appserver: abandoned call limit exceeded")
	ErrRequestLimit   = errors.New("appserver: pending server request limit exceeded")
	ErrRequestDone    = errors.New("appserver: server request is no longer pending")
)

// RPCError is a method-level failure. Data is owned JSON, and may be absent.
// Error deliberately excludes the server's message and data to avoid accidental
// disclosure through ordinary error logging. Inspect Message explicitly.
type RPCError struct {
	Code    int64           `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string { return fmt.Sprintf("appserver: RPC error %d", e.Code) }

type requestID struct {
	key string
	raw json.RawMessage
}

func parseID(raw []byte) (requestID, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return requestID{}, ErrProtocol
	}
	var key string
	if raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) != nil {
			return requestID{}, ErrProtocol
		}
		key = "s:" + s
	} else {
		n, err := strconv.ParseInt(string(raw), 10, 64)
		if err != nil {
			return requestID{}, ErrProtocol
		}
		key = "i:" + strconv.FormatInt(n, 10)
	}
	return requestID{key: key, raw: bytes.Clone(raw)}, nil
}

type envelope struct {
	id                  requestID
	hasID               bool
	method              string
	params, result, raw json.RawMessage
	rpcError            *RPCError
}

// validateJSON rejects duplicate keys at every depth, including escaped aliases.
// A bounded nesting depth also bounds recursive parser work. Payloads otherwise
// stay generic; unknown methods and envelope extension fields are preserved.
func validateJSON(raw []byte) error {
	if !utf8.Valid(raw) || !validUnicodeEscapes(raw) {
		return ErrProtocol
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var value func(int) error
	value = func(depth int) error {
		if depth > 64 {
			return ErrProtocol
		}
		t, err := d.Token()
		if err != nil {
			return ErrProtocol
		}
		if delim, ok := t.(json.Delim); ok {
			switch delim {
			case '{':
				keys := map[string]bool{}
				for d.More() {
					k, err := d.Token()
					if err != nil {
						return ErrProtocol
					}
					s, ok := k.(string)
					if !ok || keys[s] {
						return ErrProtocol
					}
					keys[s] = true
					if err := value(depth + 1); err != nil {
						return err
					}
				}
				t, err = d.Token()
				if err != nil || t != json.Delim('}') {
					return ErrProtocol
				}
			case '[':
				for d.More() {
					if err := value(depth + 1); err != nil {
						return err
					}
				}
				t, err = d.Token()
				if err != nil || t != json.Delim(']') {
					return ErrProtocol
				}
			default:
				return ErrProtocol
			}
		}
		return nil
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return ErrProtocol
	}
	return nil
}

// encoding/json replaces invalid UTF-16 escapes. Reject them before decoding
// so distinct wire IDs and keys cannot collapse onto the replacement character.
func validUnicodeEscapes(raw []byte) bool {
	inString := false
	for i := 0; i < len(raw); i++ {
		if raw[i] == '"' {
			inString = !inString
			continue
		}
		if !inString || raw[i] != '\\' {
			continue
		}
		i++
		if i >= len(raw) {
			return false
		}
		if raw[i] != 'u' {
			continue
		}
		if i+5 > len(raw) {
			return false
		}
		n, err := strconv.ParseUint(string(raw[i+1:i+5]), 16, 16)
		if err != nil {
			return false
		}
		if n >= 0xdc00 && n <= 0xdfff {
			return false
		}
		if n >= 0xd800 && n <= 0xdbff {
			if i+11 > len(raw) || raw[i+5] != '\\' || raw[i+6] != 'u' {
				return false
			}
			low, err := strconv.ParseUint(string(raw[i+7:i+11]), 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return false
			}
			i += 10
		} else {
			i += 4
		}
	}
	return true
}

func parseEnvelope(raw []byte) (envelope, error) {
	var e envelope
	if err := validateJSON(raw); err != nil {
		return e, err
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return e, ErrProtocol
	}
	var f map[string]json.RawMessage
	if json.Unmarshal(raw, &f) != nil {
		return e, ErrProtocol
	}
	id, hasID := f["id"]
	method, hasMethod := f["method"]
	result, hasResult := f["result"]
	rpcErr, hasError := f["error"]
	if hasID {
		var err error
		e.id, err = parseID(id)
		if err != nil {
			return e, err
		}
		e.hasID = true
	}
	if hasMethod {
		if hasResult || hasError || json.Unmarshal(method, &e.method) != nil || e.method == "" {
			return e, ErrProtocol
		}
		e.params = f["params"]
	} else {
		if !hasID || hasResult == hasError {
			return e, ErrProtocol
		}
		if _, ok := f["params"]; ok {
			return e, ErrProtocol
		}
		if hasResult {
			e.result = result
		} else {
			var ef map[string]json.RawMessage
			if json.Unmarshal(rpcErr, &ef) != nil || ef == nil {
				return e, ErrProtocol
			}
			code, ok := ef["code"]
			if !ok {
				return e, ErrProtocol
			}
			n, err := strconv.ParseInt(string(code), 10, 64)
			if err != nil {
				return e, ErrProtocol
			}
			msg, ok := ef["message"]
			if !ok || len(msg) == 0 || msg[0] != '"' {
				return e, ErrProtocol
			}
			var s string
			if json.Unmarshal(msg, &s) != nil {
				return e, ErrProtocol
			}
			e.rpcError = &RPCError{Code: n, Message: s, Data: ef["data"]}
		}
	}
	e.raw = bytes.Clone(raw)
	return e, nil
}

func encodeFrame(v any, limit int) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, ErrProtocol
	}
	if len(b) > limit {
		return nil, ErrFrameLimit
	}
	return append(b, '\n'), nil
}
func validPayload(raw json.RawMessage, limit int) error {
	if len(raw) > limit {
		return ErrFrameLimit
	}
	if raw != nil {
		return validateJSON(raw)
	}
	return nil
}
