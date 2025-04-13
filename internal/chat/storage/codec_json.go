package storage

import "encoding/json"

// Ensure JSONCodec implements Codec interface.
var _ Codec[any, any] = (*JSONCodec[any, any])(nil)

// JSONCodec is a codec for encoding and decoding keys and values
// using standard Go JSON serialization.
type JSONCodec[K, V any] struct{}

// EncodeKey encodes a key into a JSON byte slice for a storage backend.
func (c *JSONCodec[K, V]) EncodeKey(key K) ([]byte, error) {
	return json.Marshal(key)
}

// DecodeKey decodes a JSON byte slice into a key from a storage backend.
func (c *JSONCodec[K, V]) DecodeKey(data []byte) (K, error) {
	var key K
	if err := json.Unmarshal(data, &key); err != nil {
		return key, err
	}
	return key, nil
}

// EncodeValue encodes a value into a JSON byte slice for a storage backend.
func (c *JSONCodec[K, V]) EncodeValue(value V) ([]byte, error) {
	return json.Marshal(value)
}

// DecodeValue decodes a JSON byte slice into a value from a storage backend.
func (c *JSONCodec[K, V]) DecodeValue(data []byte) (V, error) {
	var value V
	if err := json.Unmarshal(data, &value); err != nil {
		return value, err
	}
	return value, nil
}
