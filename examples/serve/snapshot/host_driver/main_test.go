// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

func validRequest() request {
	return request{Operation: "capture", Attempt: strings.Repeat("a", 32), Artifact: strings.Repeat("b", 32),
		Container: strings.Repeat("c", 64), Image: "sha256:" + strings.Repeat("d", 64),
		Containerd: "/run/docker/containerd/containerd.sock", TimeoutSeconds: 60}
}

func TestRequestValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*request)
	}{
		{"operation", func(r *request) { r.Operation = "exec" }},
		{"attempt", func(r *request) { r.Attempt = "../../other" }},
		{"artifact", func(r *request) { r.Artifact = "" }},
		{"container", func(r *request) { r.Container = "name" }},
		{"image", func(r *request) { r.Image = "latest" }},
		{"socket", func(r *request) { r.Containerd = "/tmp/untrusted.sock" }},
		{"timeout", func(r *request) { r.TimeoutSeconds = 0 }},
		{"unbounded", func(r *request) { r.TimeoutSeconds = 7201 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := validRequest()
			tc.change(&r)
			if r.validate() == nil {
				t.Fatal("invalid request accepted")
			}
		})
	}
	if err := validRequest().validate(); err != nil {
		t.Fatal(err)
	}
}

type redirectTransport struct{ target string }

func (r redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = strings.TrimPrefix(r.target, "http://")
	return http.DefaultTransport.RoundTrip(req)
}

func TestInspectionOwnershipAndIsolation(t *testing.T) {
	for _, field := range []string{"valid", "id", "image", "label", "network", "pid", "privileged", "restart"} {
		t.Run(field, func(t *testing.T) {
			r := validRequest()
			info := containerInfo{ID: r.Container, Image: r.Image}
			info.Config.Labels = map[string]string{attemptLabel: r.Attempt}
			info.HostConfig.NetworkMode = "none"
			info.HostConfig.RestartPolicy.Name = "no"
			switch field {
			case "id":
				info.ID = strings.Repeat("e", 64)
			case "image":
				info.Image = "sha256:" + strings.Repeat("e", 64)
			case "label":
				info.Config.Labels[attemptLabel] = "other"
			case "network":
				info.HostConfig.NetworkMode = "host"
			case "pid":
				info.HostConfig.PidMode = "host"
			case "privileged":
				info.HostConfig.Privileged = true
			case "restart":
				info.HostConfig.RestartPolicy.Name = "always"
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.Method != "GET" || req.URL.Path != "/containers/"+r.Container+"/json" {
					t.Error("unexpected Docker API request")
				}
				if err := json.NewEncoder(w).Encode(info); err != nil {
					t.Error(err)
				}
			}))
			defer server.Close()
			runtime := dockerRuntime{request: r, http: &http.Client{Transport: redirectTransport{server.URL}}}
			_, err := runtime.inspect(context.Background(), r.Container)
			if (err == nil) != (field == "valid") {
				t.Fatalf("inspection error = %v", err)
			}
			if _, err := runtime.inspect(context.Background(), "other"); err == nil {
				t.Fatal("unowned target accepted")
			}
		})
	}
}

func TestConfiguration(t *testing.T) {
	if err := configuration(60).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestTargetIsSingleUse(t *testing.T) {
	root := filepath.Join(t.TempDir(), "attempts")
	if err := claimTarget(root, "container"); err != nil {
		t.Fatal(err)
	}
	if err := claimTarget(root, "container"); err == nil {
		t.Fatal("consumed target accepted")
	}
	if err := claimTarget(root, "fresh-container"); err != nil {
		t.Fatal(err)
	}
}

func TestSeccompBlocksIOUring(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy *specs.LinuxSeccomp
		valid  bool
	}{
		{"absent", nil, false},
		{"allow-all", &specs.LinuxSeccomp{DefaultAction: specs.ActAllow}, false},
		{"deny-all", &specs.LinuxSeccomp{DefaultAction: specs.ActErrno}, true},
		{"allow-one", &specs.LinuxSeccomp{DefaultAction: specs.ActErrno, Syscalls: []specs.LinuxSyscall{{Names: []string{"io_uring_setup"}, Action: specs.ActAllow}}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSeccomp(&specs.Spec{Linux: &specs.Linux{Seccomp: tc.policy}})
			if (err == nil) != tc.valid {
				t.Fatalf("got %v", err)
			}
		})
	}
}
