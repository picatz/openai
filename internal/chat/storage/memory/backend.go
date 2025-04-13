package memory

import (
	"context"
	"iter"

	"slices"

	"github.com/picatz/openai/internal/chat/storage"
)

var _ storage.Backend[string, string] = (*Backend[string, string])(nil)

type Backend[K comparable, V any] struct {
	store []storage.Entry[K, V]
}

// NewBackend creates a new in-memory storage backend, which uses a slice to store entries.
func NewBackend[K comparable, V any]() *Backend[K, V] {
	return &Backend[K, V]{}
}

// Get retrieves a value from the in-memory store by its key.
func (b *Backend[K, V]) Get(ctx context.Context, key K) (V, bool, error) {
	for _, entry := range b.store {
		if entry.Key == key {
			return entry.Value, true, nil
		}
	}
	var zero V
	return zero, false, nil
}

// Set stores a key-value pair in the in-memory store.
func (b *Backend[K, V]) Set(ctx context.Context, key K, value V) error {
	// Check if the key already exists, and if so, update the value.
	for i, entry := range b.store {
		if entry.Key == key {
			b.store[i].Value = value
			return nil
		}
	}

	// Add the new entry to the store, at the front, similar to
	// how Pebble does it, in a sense, but not really.
	b.store = append([]storage.Entry[K, V]{{Key: key, Value: value}}, b.store...)
	return nil
}

// Delete removes a key-value pair from the in-memory store by its key.
func (b *Backend[K, V]) Delete(ctx context.Context, key K) error {
	for i, entry := range b.store {
		if entry.Key == key {
			b.store = slices.Delete(b.store, i, i+1)
			break
		}
	}

	return nil
}

// List retrieves all key-value pairs from the in-memory store, with optional pagination.
func (b *Backend[K, V]) List(ctx context.Context, pageSize *int, pageToken *K) (iter.Seq2[K, V], *K, error) {
	var entries []storage.Entry[K, V]
	var nextPageToken *K

	if pageToken != nil {
		for i, entry := range b.store {
			if entry.Key == *pageToken {
				entries = b.store[i+1:]
				break
			}
		}
	} else {
		entries = b.store
	}

	if pageSize != nil && len(entries) > *pageSize {
		entries = entries[:*pageSize]
		nextPageToken = &entries[*pageSize-1].Key
	}

	return func(yield func(K, V) bool) {
		for _, entry := range entries {
			if !yield(entry.Key, entry.Value) {
				break
			}
		}
	}, nextPageToken, nil
}

// Flush is a no-op for the in-memory backend.
func (b *Backend[K, V]) Flush(context.Context) error {
	return nil
}

// Close is a no-op for the in-memory backend.
func (b *Backend[K, V]) Close(context.Context) error {
	return nil
}
