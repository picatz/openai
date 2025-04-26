package pebble

import (
	"context"
	"fmt"
	"iter"

	"github.com/cockroachdb/pebble"
	"github.com/picatz/openai/internal/chat/storage"
)

// Ensure that Backend implements the storage.Backend interface.
var _ storage.Backend[string, any] = (*Backend[string, any])(nil)

// Backend is a storage backend that uses Pebble as the underlying storage engine.
//
// Pebble can use an in-memory filesystem or a directory on disk for storage, depending
// on the options provided. By default, this application uses a directory on disk.
type Backend[K comparable, V any] struct {
	db    *pebble.DB
	codec storage.Codec[K, V]
}

// NewBackend creates a new Pebble storage backend.
func NewBackend[K comparable, V any](dirname string, opts *pebble.Options, codec storage.Codec[K, V]) (*Backend[K, V], error) {
	db, err := pebble.Open(dirname, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to open pebble database: %w", err)
	}

	return &Backend[K, V]{db: db, codec: codec}, nil
}

// Get retrieves a value from the storage backend by its key.
func (b *Backend[K, V]) Get(ctx context.Context, key K) (V, bool, error) {
	var zero V

	keyBytes, err := b.codec.EncodeKey(key)
	if err != nil {
		return zero, false, fmt.Errorf("failed to encode key: %w", err)
	}

	valueBytes, closer, err := b.db.Get(keyBytes)
	if err != nil {
		return zero, false, nil
	}
	defer closer.Close()

	value, err := b.codec.DecodeValue(valueBytes)
	if err != nil {
		return zero, false, fmt.Errorf("failed to decode value: %w", err)
	}

	return value, true, nil
}

// Set stores a key-value pair in the storage backend.
func (b *Backend[K, V]) Set(ctx context.Context, key K, value V) error {
	keyBytes, err := b.codec.EncodeKey(key)
	if err != nil {
		return fmt.Errorf("failed to encode key: %w", err)
	}

	valueBytes, err := b.codec.EncodeValue(value)
	if err != nil {
		return fmt.Errorf("failed to encode value: %w", err)
	}

	if err := b.db.Set(keyBytes, valueBytes, nil); err != nil {
		return fmt.Errorf("failed to set value: %w", err)
	}

	return nil
}

// Delete removes a key-value pair from the storage backend.
func (b *Backend[K, V]) Delete(ctx context.Context, key K) error {
	keyBytes, err := b.codec.EncodeKey(key)
	if err != nil {
		return fmt.Errorf("failed to encode key: %w", err)
	}

	if err := b.db.Delete(keyBytes, nil); err != nil {
		return fmt.Errorf("failed to delete key: %w", err)
	}

	return nil
}

// DefaultListPageSize is the default page size for listing items.
const DefaultListPageSize = 25

// List retrieves a list of key-value pairs from the storage backend.
func (b *Backend[K, V]) List(ctx context.Context, pageSize *int, pageToken *K) (iter.Seq2[K, V], *K, error) {
	iterOpts := &pebble.IterOptions{}

	if pageToken != nil {
		lowerBoundKey, err := b.codec.EncodeKey(*pageToken)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to encode pebble storage backend lower bound key: %w", err)
		}

		iterOpts.LowerBound = lowerBoundKey
	}

	listLimit := DefaultListPageSize
	if pageSize != nil {
		listLimit = *pageSize
	}

	iter, err := b.db.NewIter(iterOpts)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create pebbled storage backend iterator: %w", err)
	}
	defer iter.Close()

	var (
		values        []storage.Entry[K, V]
		nextPageToken *K
	)

	for iter.First(); iter.Valid(); iter.Next() {
		if ctx.Err() != nil {
			return func(yield func(K, V) bool) {
				for _, v := range values {
					if !yield(v.Key, v.Value) {
						break
					}
				}
			}, nil, fmt.Errorf("stopped iteration via context: %w", ctx.Err())
		}

		k, err := b.codec.DecodeKey(iter.Key())
		if err != nil {
			return nil, nil, fmt.Errorf("failed to decode key: %w", err)
		}

		v, err := b.codec.DecodeValue(iter.Value())
		if err != nil {
			return nil, nil, fmt.Errorf("failed to decode value: %w", err)
		}

		values = append(values, storage.Entry[K, V]{Key: k, Value: v})

		if len(values) >= listLimit {
			if iter.Next() {
				nextKey, err := b.codec.DecodeKey(iter.Key())
				if err != nil {
					return nil, nil, fmt.Errorf("failed to decode next key: %w", err)
				}
				nextPageToken = &nextKey
			}
			break
		}
	}
	if iter.Error() != nil {
		return nil, nil, fmt.Errorf("failed to list items: %w", iter.Error())
	}

	return func(yield func(K, V) bool) {
		for _, v := range values {
			if !yield(v.Key, v.Value) {
				break
			}
		}
	}, nextPageToken, nil
}

// Flush flushes the storage backend.
func (b *Backend[K, V]) Flush(ctx context.Context) error {
	if err := b.db.Flush(); err != nil {
		return fmt.Errorf("failed to flush pebble database: %w", err)
	}
	return nil
}

// Close closes the storage backend.
func (b *Backend[K, V]) Close(ctx context.Context) error {
	if err := b.db.Close(); err != nil {
		return fmt.Errorf("failed to close pebble database: %w", err)
	}
	return nil
}
