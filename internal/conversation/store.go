package conversation

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"time"
)

const maxSessionBytes = 8 << 20

var validID = regexp.MustCompile(`^[0-9A-Za-z]{27}$`)
var ErrConflict = errors.New("session changed in another process; reload it before saving")

type Store struct{ Dir string }

func (s Store) path(id string) (string, error) {
	if !validID.MatchString(id) {
		return "", fmt.Errorf("invalid session ID")
	}
	return filepath.Join(s.Dir, id+".json"), nil
}
func (s Store) Load(id string) (Session, error) {
	var session Session
	path, err := s.path(id)
	if err != nil {
		return session, err
	}
	dirInfo, err := os.Lstat(s.Dir)
	if err != nil {
		return session, err
	}
	if !dirInfo.IsDir() {
		return session, fmt.Errorf("history directory must not be a symlink or file")
	}
	info, err := os.Lstat(path)
	if err != nil {
		return session, err
	}
	if !info.Mode().IsRegular() {
		return session, fmt.Errorf("session must be a regular file")
	}
	f, err := os.Open(path)
	if err != nil {
		return session, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxSessionBytes+1))
	if err != nil {
		return session, err
	}
	if len(data) > maxSessionBytes {
		return session, fmt.Errorf("session exceeds 8 MiB")
	}
	if err := json.Unmarshal(data, &session); err != nil {
		return session, fmt.Errorf("decode session %s: %w", id, err)
	}
	if session.Version != 1 || session.ID != id || session.Revision < 1 || (session.API != Chat && session.API != Responses) || session.Model == "" {
		return Session{}, fmt.Errorf("invalid or unsupported session %s", id)
	}
	for _, m := range session.Messages {
		if m.Role != "system" && m.Role != "user" && m.Role != "assistant" {
			return Session{}, fmt.Errorf("unsupported message role %q", m.Role)
		}
	}
	return session, nil
}
func (s Store) List() ([]Session, error) {
	info, err := os.Lstat(s.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("history directory must not be a symlink or file")
	}
	entries, err := os.ReadDir(s.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var sessions []Session
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		id := entry.Name()[:len(entry.Name())-5]
		session, err := s.Load(id)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}
	sort.Slice(sessions, func(i, j int) bool { return sessions[i].Updated.After(sessions[j].Updated) })
	return sessions, nil
}

// Save is atomic and checks the loaded revision under an exclusive lock. An
// interrupted write leaves the previous history intact; a concurrent writer is
// rejected instead of silently replacing newer history.
func (s Store) Save(session *Session) error {
	path, err := s.path(session.ID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.Dir, 0700); err != nil {
		return err
	}
	info, err := os.Lstat(s.Dir)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("history directory must not be a symlink or file")
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("session is locked by another writer; if no CLI is running, remove %s.lock", session.ID)
		}
		return err
	}
	defer func() { lock.Close(); os.Remove(path + ".lock") }()
	current, err := s.Load(session.ID)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err == nil && current.Revision != session.Revision || errors.Is(err, os.ErrNotExist) && session.Revision != 0 {
		return ErrConflict
	}
	next := session.Clone()
	next.Revision++
	next.Updated = time.Now().UTC()
	data, err := json.MarshalIndent(next, "", "  ")
	if err != nil {
		return err
	}
	if len(data) > maxSessionBytes {
		return fmt.Errorf("session exceeds 8 MiB")
	}
	f, err := os.CreateTemp(s.Dir, ".session-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return err
	}
	*session = next
	return nil
}
