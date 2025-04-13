package storage

import (
	"context"
	"iter"
)

type Entry[K, V any] struct {
	Key   K
	Value V
}

type Backend[K, V any] interface {
	Get(ctx context.Context, key K) (value V, found bool, err error)
	Set(ctx context.Context, key K, value V) error
	Delete(ctx context.Context, key K) error
	List(ctx context.Context, pageSize *int, pageToken *K) (entries iter.Seq2[K, V], nextPageToken *K, err error)
	Flush(ctx context.Context) error
	Close(ctx context.Context) error
}

func ptr[T any](v T) *T {
	return &v
}

func PageSize(pageSize int) *int {
	return ptr(pageSize)
}

func PageToken[T any](pageToken T) *T {
	return ptr(pageToken)
}
