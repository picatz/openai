package storage

// Codec is an interface for encoding and decoding keys and values
// for a storage backend. This could be a JSON codec, a binary codec,
// or any other serialization format that makes sense for your application.
type Codec[K, V any] interface {
	EncodeKey(K) ([]byte, error)
	DecodeKey([]byte) (K, error)
	EncodeValue(V) ([]byte, error)
	DecodeValue([]byte) (V, error)
}
